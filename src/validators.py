"""Two-layer validation backstop for pipeline proposals.

Layer 1 (prompt injection) happens inside ``pipelines.py`` — the model sees
glossary entries and v2 clauses at generation time, preventing most errors.

Layer 2 (this module) catches what generation missed:

- ``validate_reference``  — does ``clause_reference`` exist in v2?
- ``validate_entailment`` — is ``new_text`` supported by the cited clause?
- ``validate_glossary``   — are equipment IDs preserved / migrated correctly?

``validate_all()`` orchestrates all three over saved pipeline results and
writes ``validation_results.json``.

Usage
-----
    python src/validators.py results/improved
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel

from llm import STRONG_MODEL, call_llm
from schemas import (
    EditProposal,
    EditProposalList,
    ProposalAction,
    Standard,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# 1. Reference validator — deterministic, pure
# ---------------------------------------------------------------------------


def validate_reference(proposal: EditProposal, v2_clause_ids: set[str]) -> bool:
    """Return True if ``proposal.clause_reference`` exists in v2.

    The caller is responsible for filtering proposals that should not be
    checked (``clause_reference="0.0.0"`` or ``action="flag"``).
    """
    return proposal.clause_reference in v2_clause_ids


# ---------------------------------------------------------------------------
# 2. Entailment validator — LLM-as-judge
# ---------------------------------------------------------------------------

# Duplicated from eval.py by design — eval.py was written and tested first;
# touching it to extract shared code is unnecessary risk.

_AIQS_REF_RE = re.compile(r"\bAIQS\s+\d+\.\d+\.\d+\b")


def _is_citation_only_change(old_text: str, new_text: str) -> bool:
    """True if old_text and new_text differ only in AIQS clause numbers."""
    old_norm = _AIQS_REF_RE.sub("AIQS_REF", old_text).strip().rstrip(".")
    new_norm = _AIQS_REF_RE.sub("AIQS_REF", new_text).strip().rstrip(".")
    return old_norm == new_norm


class _EntailmentJudgment(BaseModel):
    entailed: bool
    rationale: str


_ENTAILMENT_SYSTEM = """\
You are an expert reviewer evaluating whether a proposed edit to a Work \
Instruction is faithfully supported by the cited clause from a quality standard.

A proposal is "entailed" if the new text:
- aligns with the requirements stated in the clause, and
- does not introduce claims, procedures, or constraints absent from the clause.

A proposal is "not entailed" if the new text contradicts the clause, invents \
requirements not present, or distorts the clause's meaning.

Respond with a structured judgment."""


def validate_entailment(
    proposal: EditProposal,
    v2: Standard,
) -> bool:
    """Return True if ``new_text`` is entailed by the cited v2 clause.

    Skips citation-only changes (those are covered by ``validate_reference``).
    Returns False as a safety net when the LLM response cannot be parsed.

    The caller is responsible for filtering proposals that should not be
    checked (``clause_reference="0.0.0"`` or ``action="flag"``).
    """
    clause = v2.clause_by_id(proposal.clause_reference)
    if clause is None:
        return False

    if _is_citation_only_change(proposal.old_text, proposal.new_text):
        return True  # nothing substantive to judge

    try:
        user = (
            f"Clause from v2 of the standard:\n\n"
            f'"{clause.body}"\n\n'
            f"The pipeline proposed updating Work Instruction text:\n\n"
            f'FROM: "{proposal.old_text}"\n'
            f'TO:   "{proposal.new_text}"\n\n'
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
        judgment = resp.choices[0].message.parsed
        return judgment.entailed
    except Exception:
        return False  # safety net — on failure, reject


# ---------------------------------------------------------------------------
# 3. Glossary validator — deterministic, pure
# ---------------------------------------------------------------------------


def validate_glossary(
    proposal: EditProposal,
    glossary: list[dict],
) -> dict[str, bool]:
    """Check equipment-ID compliance in ``new_text`` only.

    Returns
    -------
    dict with two keys:
        ``preservation`` — True if every non-superseded equipment ID that
            appears in ``new_text`` is in the exact canonical format.
        ``migration`` — True if every superseded equipment ID in
            ``new_text`` has been correctly replaced with its successor.

    Both default to True when no relevant IDs are found (nothing to check).
    """
    equipment = [g for g in glossary if g.get("category") == "equipment_id"]

    preservation_ok = True
    migration_ok = True

    for eq in equipment:
        term = eq["term"]
        superseded_by = eq.get("superseded_by")

        if superseded_by:
            # Migration check: the OLD id should NOT appear in new_text;
            # the NEW id should appear if the equipment is referenced.
            if term in proposal.new_text:
                migration_ok = False
        else:
            # Preservation check: if this ID appears, it must be exact.
            # Check common_llm_errors to detect mangled versions.
            for error_form in eq.get("common_llm_errors", []):
                if error_form in proposal.new_text and term not in proposal.new_text:
                    preservation_ok = False

    return {"preservation": preservation_ok, "migration": migration_ok}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def validate_all(
    results_dir: Path,
    v2: Standard,
    glossary: list[dict],
) -> list[dict]:
    """Run all three validators over saved pipeline results.

    Parameters
    ----------
    results_dir : Path
        Directory containing ``WI-*.json`` pipeline outputs.
    v2 : Standard
        The v2 standard (for reference and entailment checks).
    glossary : list[dict]
        Glossary entries (for terminology checks).

    Returns
    -------
    list[dict]
        One entry per proposal with validation results.
    """
    v2_ids = {c.clause_id for c in v2.all_clauses()}
    results: list[dict] = []

    wi_files = sorted(results_dir.glob("WI-*.json"))
    if not wi_files:
        print(f"No WI-*.json files found in {results_dir}", file=sys.stderr)
        return results

    for wi_path in wi_files:
        wi_id = wi_path.stem
        raw = json.loads(wi_path.read_text(encoding="utf-8"))
        proposal_list = EditProposalList.model_validate(raw)

        for proposal in proposal_list.edits:
            entry: dict = {
                "wi_id": wi_id,
                "clause_reference": proposal.clause_reference,
                "action": proposal.action.value,
            }

            # Filter: skip proposals that validators should not check
            skip = (
                proposal.clause_reference == "0.0.0"
                or proposal.action == ProposalAction.FLAG
            )

            if skip:
                entry["ref_valid"] = None
                entry["entailment_valid"] = None
                entry["glossary_check"] = None
            else:
                entry["ref_valid"] = validate_reference(proposal, v2_ids)
                entry["entailment_valid"] = validate_entailment(proposal, v2)
                entry["glossary_check"] = validate_glossary(proposal, glossary)

            results.append(entry)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run validators over saved pipeline results."
    )
    ap.add_argument(
        "results_dir",
        type=Path,
        help="Directory with WI-*.json pipeline outputs (e.g. results/improved)",
    )
    args = ap.parse_args()

    if not args.results_dir.exists():
        sys.exit(f"Results directory not found: {args.results_dir}")

    # Load corpus
    v2 = Standard.model_validate(
        json.loads(
            (_DATA_DIR / "standards" / "acme_qs_v2.json").read_text(encoding="utf-8")
        )
    )
    glossary = json.loads(
        (_DATA_DIR / "glossary.json").read_text(encoding="utf-8")
    )

    print(f"Validating proposals in {args.results_dir} ...", file=sys.stderr)
    results = validate_all(args.results_dir, v2, glossary)

    # Summary
    checked = [r for r in results if r["ref_valid"] is not None]
    ref_failures = sum(1 for r in checked if not r["ref_valid"])
    ent_failures = sum(1 for r in checked if not r["entailment_valid"])
    glos_failures = sum(
        1 for r in checked
        if r["glossary_check"] and not all(r["glossary_check"].values())
    )

    print(f"\nValidated {len(checked)} proposals ({len(results) - len(checked)} skipped)")
    print(f"  Reference failures:  {ref_failures}")
    print(f"  Entailment failures: {ent_failures}")
    print(f"  Glossary failures:   {glos_failures}")

    # Write output
    out_path = args.results_dir / "validation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
