"""Eval harness for pipeline outputs.

Reads pipeline outputs saved by ``run_pipeline.py`` and computes:

- M0 — Retrieval recall @1, @3, @5 (gate)
- M1 — Reference hallucination rate
- M2 — Substantive hallucination rate (LLM-as-judge with STRONG_MODEL)
- M3 — Lost-in-middle (recall by position bucket)
- M4 — Rule consistency (action rate by transformation type)
- M5 — Terminology compliance (preservation / migration / expansion)
- Deprecated handling (correctly flagged / wrongly edited / missed)
- Ops (cost, tokens, latency)

Outputs:
    results/<pipeline>/metrics.json
    results/<pipeline>/metrics.md
    results/<pipeline>/_entailment_judgments.json  (when M2 runs)

Usage
-----
    python src/eval.py baseline
    python src/eval.py baseline --skip-m2     # skip the LLM-as-judge step
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from chunker import chunk_wi
from llm import STRONG_MODEL, call_llm, get_call_log, reset_call_log
from retriever import ClauseRetriever
from schemas import (
    EditProposal,
    EditProposalList,
    ProposalAction,
    Standard,
    TransformationLog,
    TransformationType,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_RESULTS_ROOT = _REPO_ROOT / "results"

GATE_RECALL_AT_5 = 0.85

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_v2() -> Standard:
    return Standard.model_validate(
        _load_json(_DATA_DIR / "standards" / "acme_qs_v2.json")
    )


def _load_transformation_log() -> TransformationLog:
    return TransformationLog.model_validate(
        _load_json(_DATA_DIR / "transformation_log.json")
    )


def _load_expected_edits() -> list[dict]:
    return _load_json(_DATA_DIR / "ground_truth" / "expected_edits.json")


def _load_glossary() -> list[dict]:
    return _load_json(_DATA_DIR / "glossary.json")


def _load_wi_markdown(wi_id: str) -> str:
    return (
        _DATA_DIR / "work_instructions" / f"{wi_id}.md"
    ).read_text(encoding="utf-8")


def _load_pipeline_output(out_dir: Path, wi_id: str) -> EditProposalList | None:
    path = out_dir / f"{wi_id}.json"
    if not path.exists():
        return None
    return EditProposalList.model_validate(_load_json(path))


def _processed_wi_ids(out_dir: Path) -> list[str]:
    return sorted(p.stem for p in out_dir.glob("WI-*.json"))


# ---------------------------------------------------------------------------
# Shared logic: v1 → v2 mapping, proposal lookup
# ---------------------------------------------------------------------------


def _v2_clause_ids(v2: Standard) -> set[str]:
    return {c.clause_id for c in v2.all_clauses()}


def _deprecated_v1_ids(log: TransformationLog) -> set[str]:
    out: set[str] = set()
    for entry in log.entries:
        if entry.transformation_type == TransformationType.DEPRECATE:
            out.update(entry.v1_clause_ids)
    return out


def _v2_targets_for_v1(v1_clause_id: str, log: TransformationLog) -> list[str]:
    """Return list of v2 clause IDs that supersede a given v1 clause ID.

    - Unchanged clauses (not in any mechanical log entry): return [v1_clause_id].
    - Renumbered / strengthened / etc.: return the entry's v2_clause_ids.
    - Deprecated: return [].
    - Split: return the multiple v2 IDs.
    """
    for entry in log.entries:
        if entry.is_semantic:
            continue
        if v1_clause_id in entry.v1_clause_ids:
            return list(entry.v2_clause_ids)
    return [v1_clause_id]


def _find_proposal(
    proposals: list[EditProposal],
    v1_clause_id: str,
    v2_targets: list[str],
) -> EditProposal | None:
    """Return the proposal matching this expected entry, or None.

    Match if proposal.clause_reference is in v2_targets (for non-deprecated
    cases) OR equals v1_clause_id (for deprecated, where v2_targets is empty).
    """
    candidates = set(v2_targets) if v2_targets else {v1_clause_id}
    for p in proposals:
        if p.clause_reference in candidates:
            return p
    return None


# ---------------------------------------------------------------------------
# M0 — Retrieval recall@k
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    n_queries: int

    @property
    def gate_pass(self) -> bool:
        return self.recall_at_5 >= GATE_RECALL_AT_5


def _find_chunk_for_ref(wi_md: str, v1_clause_id: str) -> str | None:
    """Return the text of the chunk that contains the AIQS citation, if any."""
    chunks = chunk_wi(wi_md)
    pattern = re.compile(rf"\bAIQS\s+{re.escape(v1_clause_id)}\b")
    for chunk in chunks:
        if pattern.search(chunk.text):
            return chunk.text
    return None


def compute_retrieval_recall(
    expected: list[dict],
    v2: Standard,
    log: TransformationLog,
) -> RetrievalResult:
    retriever = ClauseRetriever(v2)
    hits_at_1 = hits_at_3 = hits_at_5 = 0
    n = 0

    edit_required = [
        e for e in expected if e["expected_behavior"] == "edit_required"
    ]
    wi_md_cache: dict[str, str] = {}

    for entry in edit_required:
        wi_id = entry["wi_id"]
        v1_id = entry["clause_id"]
        v2_targets = set(_v2_targets_for_v1(v1_id, log))
        if not v2_targets:
            continue  # deprecated — excluded from retrieval recall

        if wi_id not in wi_md_cache:
            wi_md_cache[wi_id] = _load_wi_markdown(wi_id)
        chunk_text = _find_chunk_for_ref(wi_md_cache[wi_id], v1_id)
        if chunk_text is None:
            continue  # no chunk contains this reference; skip

        results = retriever.retrieve(chunk_text, k=5)
        retrieved_ids = [c.clause_id for c, _ in results]

        n += 1
        if v2_targets & set(retrieved_ids[:1]):
            hits_at_1 += 1
        if v2_targets & set(retrieved_ids[:3]):
            hits_at_3 += 1
        if v2_targets & set(retrieved_ids[:5]):
            hits_at_5 += 1

    return RetrievalResult(
        recall_at_1=hits_at_1 / n if n else 0.0,
        recall_at_3=hits_at_3 / n if n else 0.0,
        recall_at_5=hits_at_5 / n if n else 0.0,
        n_queries=n,
    )


# ---------------------------------------------------------------------------
# M1 — Reference hallucination
# ---------------------------------------------------------------------------


@dataclass
class M1Result:
    rate: float
    n_proposals: int
    n_hallucinated: int


def compute_reference_hallucination(
    all_proposals: list[EditProposal],
    v2: Standard,
    log: TransformationLog,
) -> M1Result:
    """M1 measures: of the EDIT proposals that claim a specific v2 clause,
    how many cite a clause_reference that doesn't exist in v2?

    Excluded from the denominator:
    - FLAG proposals (they don't claim a v2 clause exists — they say
      "I couldn't resolve this, human review needed").
    - Terminology-scan proposals with clause_reference="0.0.0" (no clause
      claim at all — these are equipment-ID replacements).
    """
    valid_v2 = _v2_clause_ids(v2)

    candidates = [
        p for p in all_proposals
        if p.action == ProposalAction.EDIT and p.clause_reference != "0.0.0"
    ]

    n_hallucinated = sum(
        1 for p in candidates if p.clause_reference not in valid_v2
    )

    return M1Result(
        rate=n_hallucinated / len(candidates) if candidates else 0.0,
        n_proposals=len(candidates),
        n_hallucinated=n_hallucinated,
    )


# ---------------------------------------------------------------------------
# M2 — Substantive hallucination (LLM-as-judge)
# ---------------------------------------------------------------------------


class _EntailmentJudgment(BaseModel):
    entailed: bool
    rationale: str


@dataclass
class M2Result:
    rate: float
    n_evaluated: int
    n_not_entailed: int
    skipped: bool = False
    n_excluded_citation_only: int = 0
    details: list[dict] = field(default_factory=list)


_ENTAILMENT_SYSTEM = """You are an expert reviewer evaluating whether a \
proposed edit to a Work Instruction is faithfully supported by the cited \
clause from a quality standard.

A proposal is "entailed" if the new text:
- aligns with the requirements stated in the clause, and
- does not introduce claims, procedures, or constraints absent from the clause.

A proposal is "not entailed" if the new text contradicts the clause, invents \
requirements not present, or distorts the clause's meaning.

Respond with a structured judgment."""

# Regex to normalise away AIQS clause numbers when detecting citation-only changes
_AIQS_REF_RE = re.compile(r"\bAIQS\s+\d+\.\d+\.\d+\b")


def _is_citation_only_change(old_text: str, new_text: str) -> bool:
    """True if old_text and new_text differ only in their AIQS clause numbers.

    Citation-only changes (e.g. "as per AIQS 3.4.4" → "as per AIQS 3.4.3")
    are excluded from M2 because the judge cannot evaluate them by reading
    the clause body alone — they are covered by M1 (does the new ID exist?)
    and M4 (was the right transformation type applied?).
    """
    old_norm = _AIQS_REF_RE.sub("AIQS_REF", old_text).strip().rstrip(".")
    new_norm = _AIQS_REF_RE.sub("AIQS_REF", new_text).strip().rstrip(".")
    return old_norm == new_norm


def _judge_entailment(
    clause_body: str, old_text: str, new_text: str
) -> _EntailmentJudgment:
    user = (
        f"Clause from v2 of the standard:\n\n"
        f'"{clause_body}"\n\n'
        f"The pipeline proposed updating Work Instruction text:\n\n"
        f'FROM: "{old_text}"\n'
        f'TO:   "{new_text}"\n\n'
        f"Is the new text faithfully supported by the clause?"
    )
    resp = call_llm(
        messages=[
            {"role": "system", "content": _ENTAILMENT_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=STRONG_MODEL,
        temperature=0.0,
        response_format=_EntailmentJudgment,
    )
    return resp.choices[0].message.parsed


def compute_substantive_hallucination(
    all_proposals: list[EditProposal],
    v2: Standard,
    log: TransformationLog,
    skip: bool = False,
) -> M2Result:
    if skip:
        return M2Result(rate=0.0, n_evaluated=0, n_not_entailed=0, skipped=True)

    valid_v2 = _v2_clause_ids(v2)
    deprecated_v1 = _deprecated_v1_ids(log)
    n_not = 0
    n_eval = 0
    n_excluded = 0
    details: list[dict] = []

    for p in all_proposals:
        if p.action != ProposalAction.EDIT:
            continue
        is_valid = (
            p.clause_reference in valid_v2
            or p.clause_reference in deprecated_v1
        )
        if not is_valid:
            continue
        clause = v2.clause_by_id(p.clause_reference)
        if clause is None:
            continue

        # Skip citation-only changes — judge cannot evaluate these meaningfully
        if _is_citation_only_change(p.old_text, p.new_text):
            n_excluded += 1
            continue

        n_eval += 1
        try:
            judgment = _judge_entailment(clause.body, p.old_text, p.new_text)
            details.append(
                {
                    "clause_reference": p.clause_reference,
                    "old_text": p.old_text,
                    "new_text": p.new_text,
                    "entailed": judgment.entailed,
                    "rationale": judgment.rationale,
                }
            )
            if not judgment.entailed:
                n_not += 1
        except Exception as exc:
            print(
                f"  ! entailment judge failed for {p.clause_reference}: {exc}",
                file=sys.stderr,
            )

    return M2Result(
        rate=n_not / n_eval if n_eval else 0.0,
        n_evaluated=n_eval,
        n_not_entailed=n_not,
        n_excluded_citation_only=n_excluded,
        details=details,
    )


# ---------------------------------------------------------------------------
# Match table — used by M3, M4, deprecated_handling
# ---------------------------------------------------------------------------


@dataclass
class MatchedEntry:
    expected: dict
    proposal: EditProposal | None  # None = pipeline silenced


def _build_match_table(
    expected: list[dict],
    proposals_by_wi: dict[str, list[EditProposal]],
    log: TransformationLog,
) -> list[MatchedEntry]:
    matches: list[MatchedEntry] = []
    for entry in expected:
        wi_id = entry["wi_id"]
        v1_id = entry["clause_id"]
        v2_targets = _v2_targets_for_v1(v1_id, log)
        proposal = _find_proposal(
            proposals_by_wi.get(wi_id, []), v1_id, v2_targets
        )
        matches.append(MatchedEntry(expected=entry, proposal=proposal))
    return matches


# ---------------------------------------------------------------------------
# M3 — Lost-in-middle
# ---------------------------------------------------------------------------


@dataclass
class M3Result:
    recall_top: float
    recall_middle: float
    recall_bottom: float
    n_top: int
    n_middle: int
    n_bottom: int

    @property
    def gap(self) -> float:
        return max(self.recall_top, self.recall_bottom) - self.recall_middle


def compute_lost_in_middle(matches: list[MatchedEntry]) -> M3Result:
    bucket_total: dict[str, int] = defaultdict(int)
    bucket_hits: dict[str, int] = defaultdict(int)

    for m in matches:
        if m.expected["expected_behavior"] not in ("edit_required", "flag_for_review"):
            continue
        bucket = m.expected["position_bucket"]
        bucket_total[bucket] += 1
        if m.proposal is not None:
            bucket_hits[bucket] += 1

    def _r(b: str) -> float:
        t = bucket_total[b]
        return bucket_hits[b] / t if t else 0.0

    return M3Result(
        recall_top=_r("top"),
        recall_middle=_r("middle"),
        recall_bottom=_r("bottom"),
        n_top=bucket_total["top"],
        n_middle=bucket_total["middle"],
        n_bottom=bucket_total["bottom"],
    )


# ---------------------------------------------------------------------------
# M4 — Rule consistency
# ---------------------------------------------------------------------------


@dataclass
class M4Result:
    by_type: dict[str, dict[str, float | int]]
    std_dev: float
    min_rate: float


def compute_rule_consistency(matches: list[MatchedEntry]) -> M4Result:
    by_type_total: dict[str, int] = defaultdict(int)
    by_type_hits: dict[str, int] = defaultdict(int)

    for m in matches:
        if m.expected["expected_behavior"] != "edit_required":
            continue
        ttype = m.expected.get("transformation_type")
        if ttype is None:
            continue
        by_type_total[ttype] += 1
        if m.proposal is not None:
            by_type_hits[ttype] += 1

    # Deprecated entries count under rule consistency too
    for m in matches:
        if m.expected["expected_behavior"] == "flag_for_review":
            ttype = m.expected.get("transformation_type")
            if ttype == "deprecate":
                by_type_total["deprecate"] += 1
                if m.proposal is not None:
                    by_type_hits["deprecate"] += 1

    by_type = {
        t: {
            "rate": by_type_hits[t] / by_type_total[t] if by_type_total[t] else 0.0,
            "n": by_type_total[t],
        }
        for t in sorted(by_type_total)
    }
    rates = [v["rate"] for v in by_type.values()]
    std = statistics.pstdev(rates) if len(rates) > 1 else 0.0
    return M4Result(
        by_type=by_type,
        std_dev=std,
        min_rate=min(rates) if rates else 0.0,
    )


# ---------------------------------------------------------------------------
# Deprecated handling
# ---------------------------------------------------------------------------


@dataclass
class DeprecatedResult:
    n_deprecated: int
    correctly_flagged: int
    incorrectly_edited: int
    missed: int


def compute_deprecated_handling(matches: list[MatchedEntry]) -> DeprecatedResult:
    flagged = edited = missed = total = 0
    for m in matches:
        if m.expected["expected_behavior"] != "flag_for_review":
            continue
        if m.expected.get("transformation_type") != "deprecate":
            continue
        total += 1
        if m.proposal is None:
            missed += 1
        elif m.proposal.action == ProposalAction.FLAG:
            flagged += 1
        else:
            edited += 1

    return DeprecatedResult(
        n_deprecated=total,
        correctly_flagged=flagged,
        incorrectly_edited=edited,
        missed=missed,
    )


# ---------------------------------------------------------------------------
# M5 — Terminology
# ---------------------------------------------------------------------------


@dataclass
class M5Result:
    preservation_rate: float
    migration_rate: float
    expansion_accuracy: float
    n_preservation: int
    n_migration: int
    n_expansion: int


def compute_terminology(
    all_proposals: list[EditProposal], glossary: list[dict]
) -> M5Result:
    equipment_ids = [g for g in glossary if g.get("category") == "equipment_id"]
    abbrevs = [g for g in glossary if g.get("category") == "abbreviation"]

    pres_hits = pres_total = 0
    mig_hits = mig_total = 0
    exp_hits = exp_total = 0

    for p in all_proposals:
        if p.action != ProposalAction.EDIT:
            continue

        for eq in equipment_ids:
            term = eq["term"]
            superseded = eq.get("superseded_by")
            if term in p.old_text:
                if superseded:
                    mig_total += 1
                    if superseded in p.new_text:
                        mig_hits += 1
                else:
                    pres_total += 1
                    if term in p.new_text:
                        pres_hits += 1

        for ab in abbrevs:
            abbrev = ab["term"]
            canonical = ab["description"]
            for m in re.finditer(
                rf"\b{re.escape(abbrev)}\s*\(([^)]+)\)", p.new_text
            ):
                exp_total += 1
                if m.group(1).strip().lower() == canonical.lower():
                    exp_hits += 1
            for m in re.finditer(
                rf"([A-Z][\w\s]+?)\s*\(\b{re.escape(abbrev)}\b\)", p.new_text
            ):
                exp_total += 1
                if m.group(1).strip().lower() == canonical.lower():
                    exp_hits += 1

    return M5Result(
        preservation_rate=pres_hits / pres_total if pres_total else 0.0,
        migration_rate=mig_hits / mig_total if mig_total else 0.0,
        expansion_accuracy=exp_hits / exp_total if exp_total else 0.0,
        n_preservation=pres_total,
        n_migration=mig_total,
        n_expansion=exp_total,
    )


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------


def compute_ops(call_log: list[dict]) -> dict:
    by_model: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost_usd": 0.0})
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    latencies = []

    for c in call_log:
        m = c["model"]
        by_model[m]["calls"] += 1
        by_model[m]["tokens"] += c["total_tokens"]
        by_model[m]["cost_usd"] += c["cost_usd"]
        total_calls += 1
        total_tokens += c["total_tokens"]
        total_cost += c["cost_usd"]
        latencies.append(c["latency_s"])

    return {
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "by_model": {
            m: {
                "calls": d["calls"],
                "tokens": d["tokens"],
                "cost_usd": round(d["cost_usd"], 4),
            }
            for m, d in by_model.items()
        },
    }


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _build_metrics_dict(
    pipeline: str,
    n_wis: int,
    n_failed: int,
    metadata: dict,
    m0: RetrievalResult,
    m1: M1Result,
    m2: M2Result,
    m3: M3Result,
    m4: M4Result,
    m5: M5Result,
    dep: DeprecatedResult,
    ops: dict,
) -> dict:
    return {
        "pipeline": pipeline,
        "n_wis": n_wis,
        "n_wis_failed": n_failed,
        "metrics": {
            "M0_retrieval_recall": {
                "@1": round(m0.recall_at_1, 3),
                "@3": round(m0.recall_at_3, 3),
                "@5": round(m0.recall_at_5, 3),
                "gate_pass": m0.gate_pass,
                "gate_target": GATE_RECALL_AT_5,
                "n_queries": m0.n_queries,
                "applicable": pipeline == "baseline",
            },
            "M1_reference_hallucination": {
                "rate": round(m1.rate, 3),
                "n_proposals": m1.n_proposals,
                "n_hallucinated": m1.n_hallucinated,
            },
            "M2_substantive_hallucination": {
                "rate": round(m2.rate, 3),
                "n_evaluated": m2.n_evaluated,
                "n_not_entailed": m2.n_not_entailed,
                "n_excluded_citation_only": m2.n_excluded_citation_only,
                "skipped": m2.skipped,
            },
            "M3_lost_in_middle": {
                "recall_top": round(m3.recall_top, 3),
                "recall_middle": round(m3.recall_middle, 3),
                "recall_bottom": round(m3.recall_bottom, 3),
                "gap": round(m3.gap, 3),
                "n_top": m3.n_top,
                "n_middle": m3.n_middle,
                "n_bottom": m3.n_bottom,
            },
            "M4_rule_consistency": {
                "by_type": {
                    t: {"rate": round(d["rate"], 3), "n": d["n"]}
                    for t, d in m4.by_type.items()
                },
                "std_dev": round(m4.std_dev, 3),
                "min_rate": round(m4.min_rate, 3),
            },
            "M5_terminology": {
                "preservation_rate": round(m5.preservation_rate, 3),
                "migration_rate": round(m5.migration_rate, 3),
                "expansion_accuracy": round(m5.expansion_accuracy, 3),
                "n_preservation": m5.n_preservation,
                "n_migration": m5.n_migration,
                "n_expansion": m5.n_expansion,
            },
            "deprecated_handling": {
                "n_deprecated": dep.n_deprecated,
                "correctly_flagged": dep.correctly_flagged,
                "incorrectly_edited": dep.incorrectly_edited,
                "missed": dep.missed,
            },
        },
        "ops": ops,
        "metadata": metadata,
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _render_markdown(d: dict) -> str:
    m = d["metrics"]
    md = d.get("metadata", {})
    ops = d.get("ops", {})

    parts: list[str] = []
    parts.append(f"# {d['pipeline'].title()} Pipeline Evaluation")
    parts.append("")
    parts.append(
        f"Run: {md.get('timestamp', '?')} | Model: {md.get('model', '?')} | "
        f"Seed: {md.get('seed', '?')} | n_wis: {d['n_wis']}"
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # M0
    m0 = m["M0_retrieval_recall"]
    gate = "✓" if m0["gate_pass"] else "❌"
    parts.append("## Gate metric")
    parts.append("")
    parts.append("### M0 — Retrieval recall@k")
    parts.append("")

    if not m0.get("applicable", True):
        parts.append(
            "**Not applicable to the improved pipeline.** M0 measures pure "
            "embedding-search retrieval (chunk text as query → top-k v2 clauses). "
            "The improved pipeline uses hybrid retrieval: direct ID lookup for "
            "`changed` clauses, full semantic search for `id_collision` cases, "
            "and narrowed search for `v1_only` cases. The numbers below reflect "
            "baseline-style retrieval and are shown for comparability only. "
            "Retrieval effectiveness of the improved pipeline is captured "
            "indirectly by M3 (position recall) and M4 (rule consistency)."
        )
        parts.append("")

    parts.append(
        "How often the retriever returns the correct v2 clause in the top-k results, "
        "given a chunk of WI text. This is the *gate*: if retrieval is broken, no "
        "downstream metric is meaningful. Failed retrievals propagate as hallucinations "
        "in the improved pipeline."
    )
    parts.append("")
    parts.append("| k | Recall |")
    parts.append("|---|---|")
    parts.append(f"| 1 | {m0['@1']:.2f} |")
    parts.append(f"| 3 | {m0['@3']:.2f} |")
    parts.append(f"| 5 | {m0['@5']:.2f} {gate} (target ≥{m0['gate_target']}) |")
    parts.append("")
    parts.append(
        f"**n_queries:** {m0['n_queries']} (edit-requiring references; "
        "deprecated clauses excluded)"
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # M1
    parts.append("## Failure-mode metrics")
    parts.append("")
    parts.append("### M1 — Reference hallucination")
    parts.append("")
    parts.append(
        "Fraction of proposed edits citing a clause ID that does not exist in v2. "
        "Captures the PoC failure where the LLM invents AIQS X.Y.Z citations or keeps "
        "v1-only IDs that were renumbered/deprecated. Lower is better."
    )
    parts.append("")
    m1 = m["M1_reference_hallucination"]
    parts.append("| Rate | n_proposals | n_hallucinated |")
    parts.append("|---|---|---|")
    parts.append(
        f"| {_pct(m1['rate'])} | {m1['n_proposals']} | {m1['n_hallucinated']} |"
    )
    parts.append("")

    # M2
    parts.append("### M2 — Substantive hallucination")
    parts.append("")
    parts.append(
        "Fraction of proposed `new_text` strings that an LLM judge (gpt-4o, temp=0) "
        "rules as *not entailed* by the cited v2 clause. Captures the PoC failure where "
        "the pipeline writes plausible-sounding text disconnected from the standard. "
        "Hallucinated references (M1) and citation-only changes (where `old_text` and "
        "`new_text` differ only by their AIQS reference number) are excluded from the "
        "denominator — those are covered by M1 and M4. Lower is better."
    )
    parts.append("")
    m2 = m["M2_substantive_hallucination"]
    if m2.get("skipped"):
        parts.append("_Skipped (--skip-m2)_")
    else:
        parts.append(
            "| Rate | n_evaluated | n_not_entailed | n_excluded (citation-only) |"
        )
        parts.append("|---|---|---|---|")
        parts.append(
            f"| {_pct(m2['rate'])} | {m2['n_evaluated']} | "
            f"{m2['n_not_entailed']} | {m2.get('n_excluded_citation_only', 0)} |"
        )
    parts.append("")

    # M3
    parts.append("### M3 — Lost-in-middle (action recall by WI position)")
    parts.append("")
    parts.append(
        "Recall of edit/flag actions broken down by where the citation sits in the WI "
        "(top/middle/bottom thirds, normalized by section index). Captures the PoC "
        "failure where edits in the middle of long documents are missed disproportionately. "
        "Looking for: a dip in the middle bucket compared to top and bottom."
    )
    parts.append("")
    m3 = m["M3_lost_in_middle"]
    parts.append("| Bucket | Recall | n |")
    parts.append("|---|---|---|")
    parts.append(f"| Top    | {_pct(m3['recall_top'])}    | {m3['n_top']} |")
    parts.append(f"| Middle | {_pct(m3['recall_middle'])} | {m3['n_middle']} |")
    parts.append(f"| Bottom | {_pct(m3['recall_bottom'])} | {m3['n_bottom']} |")
    parts.append("")
    parts.append(
        f"**Gap (max(top,bot) − middle):** {m3['gap']:.2f} — "
        "high gap = strong lost-in-middle effect."
    )
    parts.append("")

    # M4
    parts.append("### M4 — Rule consistency (action rate by transformation type)")
    parts.append("")
    parts.append(
        "For each v1→v2 transformation type, fraction of expected actions the pipeline "
        "proposed (edit OR flag — any non-silent response). Captures the PoC failure "
        "where the same rule is applied inconsistently across different WIs. Looking "
        "for: low std_dev across types, no type below ~0.7."
    )
    parts.append("")
    m4 = m["M4_rule_consistency"]
    parts.append("| Type | Action rate | n |")
    parts.append("|---|---|---|")
    sorted_types = sorted(m4["by_type"].items(), key=lambda kv: -kv[1]["rate"])
    for t, d in sorted_types:
        parts.append(f"| {t} | {_pct(d['rate'])} | {d['n']} |")
    parts.append("")
    parts.append(
        f"**Std dev:** {m4['std_dev']:.2f} | **Min rate:** {_pct(m4['min_rate'])}"
    )
    parts.append("")

    # M5
    parts.append("### M5 — Terminology compliance")
    parts.append("")
    parts.append(
        "Three sub-metrics, computed by string matching against the glossary:\n\n"
        "- **Preservation:** when an unchanged equipment ID appears in `old_text`, "
        "the same ID appears verbatim in `new_text`. Pipeline must not mutate opaque tokens.\n"
        "- **Migration:** when a superseded equipment ID appears in `old_text`, the "
        "pipeline replaces it with the correct `superseded_by` ID in `new_text`.\n"
        "- **Expansion accuracy:** when the pipeline expands an abbreviation in "
        "`new_text` (e.g. `ORR (...)`), the expansion matches the canonical one from the glossary."
    )
    parts.append("")
    m5 = m["M5_terminology"]
    parts.append("| Sub-metric | Rate | n |")
    parts.append("|---|---|---|")
    parts.append(
        f"| Preservation | {_pct(m5['preservation_rate'])} | {m5['n_preservation']} |"
    )
    parts.append(
        f"| Migration | {_pct(m5['migration_rate'])} | {m5['n_migration']} |"
    )
    parts.append(
        f"| Expansion accuracy | {_pct(m5['expansion_accuracy'])} | {m5['n_expansion']} |"
    )
    parts.append("")

    # Deprecated
    parts.append("### Deprecated handling")
    parts.append("")
    parts.append(
        "For the v1 clauses deprecated in v2 and cited in WIs, the schema-correct "
        "action is `flag` (no concrete edit can be auto-proposed). Tracks whether "
        "the pipeline distinguishes 'edit' from 'flag' use cases."
    )
    parts.append("")
    dep = m["deprecated_handling"]
    parts.append("| Outcome | Count |")
    parts.append("|---|---|")
    parts.append(f"| Correctly flagged | {dep['correctly_flagged']} |")
    parts.append(f"| Incorrectly proposed edit | {dep['incorrectly_edited']} |")
    parts.append(f"| Missed (no action) | {dep['missed']} |")
    parts.append(f"| **Total deprecated** | **{dep['n_deprecated']}** |")
    parts.append("")
    parts.append("---")
    parts.append("")

    # Ops
    parts.append("## Ops")
    parts.append("")
    parts.append("| Metric | Value |")
    parts.append("|---|---|")
    parts.append(f"| Total cost | ${ops.get('total_cost_usd', 0):.4f} |")
    parts.append(f"| Total tokens | {ops.get('total_tokens', 0):,} |")
    parts.append(f"| Total LLM calls | {ops.get('total_calls', 0)} |")
    parts.append(f"| Avg latency per call | {ops.get('avg_latency_s', 0):.2f}s |")
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline", choices=["baseline", "improved"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--skip-m2", action="store_true", help="Skip LLM-as-judge entailment"
    )
    args = ap.parse_args()

    out_dir = args.out_dir or (_RESULTS_ROOT / args.pipeline)
    if not out_dir.exists():
        sys.exit(f"Output dir not found: {out_dir} (run run_pipeline.py first)")

    # Load corpus
    v2 = _load_v2()
    log = _load_transformation_log()
    expected = _load_expected_edits()
    glossary = _load_glossary()

    # Load pipeline outputs
    wi_ids = _processed_wi_ids(out_dir)
    if not wi_ids:
        sys.exit(f"No pipeline outputs found in {out_dir}")

    proposals_by_wi: dict[str, list[EditProposal]] = {}
    all_proposals: list[EditProposal] = []
    for wi_id in wi_ids:
        plist = _load_pipeline_output(out_dir, wi_id)
        edits = list(plist.edits) if plist else []
        proposals_by_wi[wi_id] = edits
        all_proposals.extend(edits)

    # Filter expected to processed WIs only (smoke runs don't have all 30)
    expected_filtered = [e for e in expected if e["wi_id"] in set(wi_ids)]

    # Load pipeline call log; reset so M2 entailment calls are counted separately
    pipeline_call_log = _load_json(out_dir / "_call_log.json")
    reset_call_log()

    # Compute metrics
    print("Computing M0 — retrieval recall...", file=sys.stderr)
    m0 = compute_retrieval_recall(expected_filtered, v2, log)

    print("Computing M1 — reference hallucination...", file=sys.stderr)
    m1 = compute_reference_hallucination(all_proposals, v2, log)

    print("Computing M2 — substantive hallucination...", file=sys.stderr)
    m2 = compute_substantive_hallucination(
        all_proposals, v2, log, skip=args.skip_m2
    )

    matches = _build_match_table(expected_filtered, proposals_by_wi, log)

    print("Computing M3 — lost-in-middle...", file=sys.stderr)
    m3 = compute_lost_in_middle(matches)

    print("Computing M4 — rule consistency...", file=sys.stderr)
    m4 = compute_rule_consistency(matches)

    print("Computing M5 — terminology...", file=sys.stderr)
    m5 = compute_terminology(all_proposals, glossary)

    print("Computing deprecated handling...", file=sys.stderr)
    dep = compute_deprecated_handling(matches)

    # Ops = pipeline run + entailment judge combined
    eval_call_log = get_call_log()
    ops = compute_ops(pipeline_call_log + eval_call_log)

    # Metadata from pipeline run + eval timestamp
    pipeline_meta = _load_json(out_dir / "_metadata.json")
    metadata = {
        **pipeline_meta,
        "eval_timestamp": datetime.now(timezone.utc).isoformat(),
        "skipped_m2": args.skip_m2,
        "timestamp": pipeline_meta.get("started_at"),
    }

    metrics_dict = _build_metrics_dict(
        pipeline=args.pipeline,
        n_wis=len(wi_ids),
        n_failed=pipeline_meta.get("n_wis_failed", 0),
        metadata=metadata,
        m0=m0, m1=m1, m2=m2, m3=m3, m4=m4, m5=m5,
        dep=dep, ops=ops,
    )

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_dict, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.md").write_text(
        _render_markdown(metrics_dict), encoding="utf-8"
    )

    # Save entailment judgments sidecar for manual inspection
    if not args.skip_m2 and m2.details:
        (out_dir / "_entailment_judgments.json").write_text(
            json.dumps(m2.details, indent=2), encoding="utf-8"
        )

    print()
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'metrics.md'}")
    if not args.skip_m2 and m2.details:
        print(f"Wrote {out_dir / '_entailment_judgments.json'}")


if __name__ == "__main__":
    main()
