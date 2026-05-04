"""Generate the main results comparison document.

Reads all metrics files and produces results/baseline_vs_improved.md.

Usage:
    python scripts/generate_results_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
_BASELINE_DIR = _RESULTS_DIR / "baseline"
_IMPROVED_DIR = _RESULTS_DIR / "improved"


def _load(path: Path) -> dict | list:
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _pct_int(v: float) -> str:
    return f"{v * 100:.0f}%"


def build_document() -> str:
    b = _load(_BASELINE_DIR / "metrics.json")
    i = _load(_IMPROVED_DIR / "metrics.json")
    val = _load(_IMPROVED_DIR / "validation_results.json")
    sem = _load(_IMPROVED_DIR / "semantic_eval.json")

    bm = b["metrics"]
    im = i["metrics"]
    b_ops = b.get("ops", {})
    i_ops = i.get("ops", {})

    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    # ── 1. Headline ──────────────────────────────────────────────────

    b_m1 = bm["M1_reference_hallucination"]["rate"]
    i_m1 = im["M1_reference_hallucination"]["rate"]
    i_m3_gap = im["M3_lost_in_middle"]["gap"]
    cost_ratio = b_ops.get("total_cost_usd", 1) / max(i_ops.get("total_cost_usd", 1), 0.001)

    w("# Baseline vs Improved Pipeline — Results")
    w()
    w(
        f"**TL;DR:** The improved pipeline reduces reference hallucination "
        f"from {_pct(b_m1)} to {_pct(i_m1)}, closes the lost-in-middle "
        f"position gap to {i_m3_gap:.2f}, and runs "
        f"{cost_ratio:.1f}× cheaper per WI."
    )
    w()
    w("---")
    w()

    # ── 2. Gate Metric (M0) ──────────────────────────────────────────

    b_m0 = bm["M0_retrieval_recall"]

    w("## 1. Gate Metric — Retrieval Recall (M0)")
    w()
    w(
        "Retrieval recall measures how often the retriever returns the "
        "correct v2 clause in the top-k results given a WI chunk as query. "
        "This is the foundation metric: if retrieval fails, downstream "
        "metrics are noise."
    )
    w()
    w("| k | Baseline |")
    w("|---|---|")
    w(f"| @1 | {_pct(b_m0['@1'])} |")
    w(f"| @3 | {_pct(b_m0['@3'])} |")
    w(f"| @5 | {_pct(b_m0['@5'])} |")
    w()
    w(
        "**M0 does not apply to the improved pipeline.** The improved "
        "pipeline uses hybrid retrieval: direct ID lookup for `changed` "
        "clauses (deterministically correct), narrowed semantic search for "
        "`v1_only` cases. Its retrieval effectiveness is captured indirectly "
        "by M3 (position recall) and M4 (rule consistency)."
    )
    w()
    w("---")
    w()

    # ── 3. Failure-Mode Metrics ──────────────────────────────────────

    w("## 2. Failure-Mode Metrics")
    w()

    # M1
    b_m1_d = bm["M1_reference_hallucination"]
    i_m1_d = im["M1_reference_hallucination"]

    w("### M1 — Reference Hallucination")
    w()
    w(
        "Fraction of edit proposals citing a `clause_reference` that does "
        "not exist in v2. FLAG proposals and terminology-scan proposals "
        "(`clause_reference=\"0.0.0\"`) are excluded from the denominator."
    )
    w()
    w("| | Baseline | Improved |")
    w("|---|---|---|")
    w(
        f"| Rate | {_pct(b_m1_d['rate'])} "
        f"({b_m1_d['n_hallucinated']}/{b_m1_d['n_proposals']}) | "
        f"{_pct(i_m1_d['rate'])} "
        f"({i_m1_d['n_hallucinated']}/{i_m1_d['n_proposals']}) |"
    )
    w()

    # M2
    i_m2_d = im["M2_substantive_hallucination"]
    w("### M2 — Substantive Hallucination")
    w()
    w(
        "Fraction of `new_text` strings that an independent LLM judge "
        "(gpt-4o, temperature=0) rules as not entailed by the cited v2 "
        "clause. Citation-only changes are excluded. Baseline M2 was not "
        "computed (baseline proposals are too noisy for meaningful "
        "entailment judgment)."
    )
    w()
    w("| | Improved |")
    w("|---|---|")
    if i_m2_d.get("skipped"):
        w("| Rate | _Skipped_ |")
    else:
        w(
            f"| Rate | {_pct(i_m2_d['rate'])} "
            f"({i_m2_d['n_not_entailed']}/{i_m2_d['n_evaluated']}) |"
        )
        w(
            f"| Citation-only excluded | {i_m2_d.get('n_excluded_citation_only', 0)} |"
        )
    w()

    # M3
    b_m3 = bm["M3_lost_in_middle"]
    i_m3 = im["M3_lost_in_middle"]

    w("### M3 — Lost-in-Middle (Position Recall)")
    w()
    w(
        "Edit recall bucketed by where the AIQS citation sits in the WI "
        "(top / middle / bottom third). The RAG architecture sidesteps "
        "lost-in-middle by processing each chunk independently rather "
        "than feeding the full WI to the LLM."
    )
    w()
    w("| Bucket | Baseline | Improved |")
    w("|---|---|---|")
    w(f"| Top | {_pct(b_m3['recall_top'])} (n={b_m3['n_top']}) | {_pct(i_m3['recall_top'])} (n={i_m3['n_top']}) |")
    w(f"| Middle | {_pct(b_m3['recall_middle'])} (n={b_m3['n_middle']}) | {_pct(i_m3['recall_middle'])} (n={i_m3['n_middle']}) |")
    w(f"| Bottom | {_pct(b_m3['recall_bottom'])} (n={b_m3['n_bottom']}) | {_pct(i_m3['recall_bottom'])} (n={i_m3['n_bottom']}) |")
    w(f"| **Gap (max−mid)** | **{b_m3['gap']:.2f}** | **{i_m3['gap']:.2f}** |")
    w()
    w("![Position Recall](figures/position_recall.png)")
    w()

    # M4
    b_m4 = bm["M4_rule_consistency"]
    i_m4 = im["M4_rule_consistency"]

    w("### M4 — Rule Consistency (Action Rate by Type)")
    w()
    w(
        "For each v1→v2 transformation type, the fraction of expected "
        "edits the pipeline proposed. Measures whether the same rule is "
        "applied consistently across WIs."
    )
    w()

    # Collect all types from improved (baseline may not have all)
    all_types = sorted(i_m4["by_type"].keys())
    w("| Type | Baseline | Improved |")
    w("|---|---|---|")
    for t in all_types:
        b_entry = b_m4["by_type"].get(t, {})
        i_entry = i_m4["by_type"][t]
        b_rate = _pct(b_entry["rate"]) if b_entry else "—"
        b_n = f" (n={b_entry['n']})" if b_entry else ""
        w(f"| {t} | {b_rate}{b_n} | {_pct(i_entry['rate'])} (n={i_entry['n']}) |")
    w(f"| **Std dev** | **{b_m4['std_dev']:.2f}** | **{i_m4['std_dev']:.2f}** |")
    w()
    w("![M4 by Type](figures/m4_by_type.png)")
    w()

    # M5
    b_m5 = bm["M5_terminology"]
    i_m5 = im["M5_terminology"]

    w("### M5 — Terminology Compliance")
    w()
    w(
        "Equipment-ID handling measured by string match against the glossary. "
        "Baseline produced zero proposals containing equipment IDs (the naive "
        "prompt does not surface them), so baseline rates are 0% with n=0."
    )
    w()
    w("| Sub-metric | Baseline | Improved |")
    w("|---|---|---|")
    w(
        f"| Preservation | {_pct(b_m5['preservation_rate'])} (n={b_m5['n_preservation']}) "
        f"| {_pct(i_m5['preservation_rate'])} (n={i_m5['n_preservation']}) |"
    )
    w(
        f"| Migration | {_pct(b_m5['migration_rate'])} (n={b_m5['n_migration']}) "
        f"| {_pct(i_m5['migration_rate'])} (n={i_m5['n_migration']}) |"
    )
    w()

    # Deprecated
    b_dep = bm["deprecated_handling"]
    i_dep = im["deprecated_handling"]

    w("### Deprecated Clause Handling")
    w()
    w("| Outcome | Baseline | Improved |")
    w("|---|---|---|")
    w(f"| Correctly flagged | {b_dep['correctly_flagged']}/{b_dep['n_deprecated']} | {i_dep['correctly_flagged']}/{i_dep['n_deprecated']} |")
    w(f"| Incorrectly edited | {b_dep['incorrectly_edited']} | {i_dep['incorrectly_edited']} |")
    w(f"| Missed | {b_dep['missed']} | {i_dep['missed']} |")
    w()
    w("---")
    w()

    # ── 4. Validator Layer Results ────────────────────────────────────

    checked = [r for r in val if r.get("ref_valid") is not None]
    skipped = len(val) - len(checked)
    ref_fail = sum(1 for r in checked if not r["ref_valid"])
    ent_fail = sum(1 for r in checked if not r["entailment_valid"])
    glos_fail = sum(
        1 for r in checked
        if r.get("glossary_check") and not all(r["glossary_check"].values())
    )
    total_checked = len(checked)

    w("## 3. Validator Layer Results")
    w()
    w(
        "The validator is the backstop layer: after the pipeline generates "
        "proposals (Layer 1 — prompt injection), the validator catches what "
        "generation missed. In production, proposals failing any gate would "
        "be rejected or routed to human review before reaching the end user."
    )
    w()
    w(f"**{total_checked} proposals checked** ({skipped} skipped — FLAG or terminology-scan)")
    w()
    w("| Gate | Failures | Rate |")
    w("|---|---|---|")
    w(f"| Reference (clause exists in v2) | {ref_fail} | {_pct(ref_fail/total_checked) if total_checked else '—'} |")
    w(f"| Entailment (new_text supported by clause) | {ent_fail} | {_pct(ent_fail/total_checked) if total_checked else '—'} |")
    w(f"| Glossary (equipment IDs correct) | {glos_fail} | {_pct(glos_fail/total_checked) if total_checked else '—'} |")
    w()

    pass_all = sum(
        1 for r in checked
        if r["ref_valid"]
        and r["entailment_valid"]
        and (r.get("glossary_check") is None or all(r["glossary_check"].values()))
    )
    w(
        f"**Proposals passing all three gates: {pass_all}/{total_checked} "
        f"({_pct(pass_all/total_checked) if total_checked else '—'}).** "
        f"In production, {total_checked - pass_all} proposals would be "
        f"routed to human review."
    )
    w()
    w("---")
    w()

    # ── 5. Semantic Subset ────────────────────────────────────────────

    ss = sem["summary"]
    sc = sem["by_category"]

    w("## 4. Semantic Change Subset")
    w()
    w(
        "12 hand-labeled semantic changes that computed ground truth cannot "
        "cover: tone shifts, cross-reference chain breakage, ambiguous scope "
        "changes, and clause merges. These are evaluated separately from "
        "mechanical metrics. The expected result — and the honest one — is "
        "that both pipelines perform poorly here, because semantic changes "
        "require human judgment the pipeline is not designed to provide."
    )
    w()
    w(f"**{ss['testable']} testable instances** across {len(sc)} categories "
      f"({ss['not_testable']} not testable — clause not cited by any WI)")
    w()
    w("| Category | n | Match | Wrong Action | Miss |")
    w("|---|---|---|---|---|")
    for cat, c in sc.items():
        label = cat.replace("semantic_", "")
        if c["testable"] == 0:
            w(f"| {label} | 0 | — | — | — |")
        else:
            w(f"| {label} | {c['testable']} | {c['match']} | {c['wrong_action']} | {c['miss']} |")
    w(f"| **Total** | **{ss['testable']}** | **{ss['match']}** | **{ss['wrong_action']}** | **{ss['miss']}** |")
    w()
    w("### Interpretation")
    w()
    w(
        "**0 matches is the expected result**, not a failure. Each category "
        "exposes a specific architectural limitation:"
    )
    w()
    w(
        "- **xref_chain (all miss):** The pipeline skips clauses whose text "
        "is unchanged between v1 and v2. Cross-reference chain breakage — "
        "where the clause text is identical but a referenced clause was "
        "deprecated — is invisible without a dependency graph. This is an "
        "architectural ceiling, not a bug."
    )
    w(
        "- **tone_shift / ambiguous_scope (all wrong_action):** The pipeline "
        "detects that these clauses changed and generates edits, but the "
        "ground truth says they should be flagged for human review. The "
        "pipeline has no mechanism to distinguish 'structural change → edit' "
        "from 'judgment-requiring change → flag'."
    )
    w(
        "- **clause_merge (all wrong_action):** The pipeline correctly "
        "identifies that the absorbed clause is missing from v2 and generates "
        "a flag, but the ground truth says an edit is needed (update the "
        "citation to the surviving clause). The narrowed similarity search "
        "returns low confidence for merged content, triggering a flag "
        "instead of an edit."
    )
    w()
    w("---")
    w()

    # ── 6. Operational Metrics ────────────────────────────────────────

    w("## 5. Operational Metrics")
    w()
    w("| Metric | Baseline | Improved |")
    w("|---|---|---|")
    w(f"| Total cost (pipeline) | ${b_ops.get('total_cost_usd', 0):.3f} | ${i_ops.get('total_cost_usd', 0):.3f} |")
    w(f"| Total tokens | {b_ops.get('total_tokens', 0):,} | {i_ops.get('total_tokens', 0):,} |")
    w(f"| Total LLM calls | {b_ops.get('total_calls', 0)} | {i_ops.get('total_calls', 0)} |")
    w(f"| Avg latency/call | {b_ops.get('avg_latency_s', 0):.2f}s | {i_ops.get('avg_latency_s', 0):.2f}s |")
    w(f"| Total proposals | {bm['M1_reference_hallucination']['n_proposals']} | {im['M1_reference_hallucination']['n_proposals']} |")
    w()
    w("---")
    w()

    # ── 7. Known Limitations + Proposed Improvements ──────────────────

    w("## 6. Known Limitations")
    w()

    w("### Renumber recall (37.5%)")
    w()
    w(
        "Cascading renumbers — where a clause is moved across sections — "
        "are the pipeline's weakest transformation type. The ChangeMap "
        "categorizes these as `v1_only`, and the narrowed similarity search "
        "over `v2_only` clauses often fails because the renumber target "
        "lands in `changed` (same ID exists in v2 with different content "
        "due to a cascade). Without the transformation log, the pipeline "
        "cannot distinguish a renumber from a deprecation. This is an "
        "architectural ceiling of inference-by-similarity."
    )
    w()

    w("### Deprecated clause: 1/4 missed")
    w()
    w(
        "All 4 deprecated v1 IDs are re-occupied in v2 (by renumbers or "
        "inserts), so they land in `changed` rather than `v1_only`. For "
        "3/4 cases, the content mismatch is detected and correctly flagged. "
        "For 1 case, the full v2 search finds a spurious match above the "
        "0.7 threshold, causing the LLM to generate an edit instead of "
        "a flag."
    )
    w()

    w("### M2 substantive hallucination (34%)")
    w()
    w(
        "Three known patterns account for most not-entailed judgments: "
        "equipment-ID changes attributed to the wrong clause, renumber "
        "consolidation of multiple citations, and legitimate hallucinations "
        "where the LLM made content claims not supported by the cited "
        "clause. The validator layer (Section 3) catches these at runtime."
    )
    w()

    w("### Semantic changes (0% match)")
    w()
    w(
        "The pipeline does not detect any of the 12 semantic changes "
        "correctly. See Section 4 for analysis. Three targeted improvements "
        "are proposed below."
    )
    w()

    w("---")
    w()

    # ── 8. Proposed Improvements ──────────────────────────────────────

    w("## 7. Proposed Improvements")
    w()
    w(
        "Three improvements identified from the semantic eval results, "
        "ordered by priority. None were implemented to avoid overfitting "
        "to the eval corpus."
    )
    w()

    w("### 1. Cross-reference dependency check (xref_chain)")
    w()
    w(
        "**Problem:** Pipeline skips clauses marked `unchanged` in the "
        "ChangeMap, missing cases where the clause text is identical but "
        "a referenced clause was deprecated."
    )
    w()
    w(
        "**Solution:** Before marking a clause as `unchanged`, parse its v2 "
        "body for AIQS references (`AIQS X.Y.Z`). If any referenced clause "
        "does not exist in v2, generate a `flag` proposal. Purely "
        "deterministic — regex + ChangeMap lookup, no LLM call."
    )
    w()
    w("**Effort:** Low (~1–2h). **Risk of regression:** Very low.")
    w()

    w("### 2. Change-type classifier (tone_shift, ambiguous_scope)")
    w()
    w(
        "**Problem:** For `changed` clauses, the pipeline always generates "
        "`edit`. It cannot distinguish structural changes (→ edit) from "
        "judgment-requiring changes (→ flag for review)."
    )
    w()
    w(
        "**Solution:** Add a classification step before generation. Given "
        "(v1_body, v2_body), a lightweight LLM call decides whether the "
        "change is structural or judgment-requiring. Signals: hedging "
        "removal without structural change → flag; qualifier replacement "
        "(precise → vague) → flag; new obligation or requirement → edit."
    )
    w()
    w(
        "**Effort:** Medium (~2–3h). **Risk of regression:** Medium (adds "
        "one LLM call per `changed` clause, ~$0.01 additional per run)."
    )
    w()

    w("### 3. Merge detector (clause_merge)")
    w()
    w(
        "**Problem:** Absorbed clauses in merges land in `v1_only`. The "
        "narrowed search returns low similarity (< 0.7) because the "
        "content was redistributed, triggering a flag instead of an edit."
    )
    w()
    w(
        "**Solution:** Lower the similarity threshold to ~0.4 for "
        "`v1_only` cases. If any match is found above this relaxed "
        "threshold, send to the LLM with an instruction to propose a "
        "citation update rather than flagging. Calibration needed to "
        "avoid false positives on genuinely deprecated clauses."
    )
    w()
    w("**Effort:** Low (~1h). **Risk of regression:** Medium (threshold tuning).")
    w()

    return "\n".join(lines)


def main() -> None:
    doc = build_document()
    out_path = _RESULTS_DIR / "baseline_vs_improved.md"
    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
