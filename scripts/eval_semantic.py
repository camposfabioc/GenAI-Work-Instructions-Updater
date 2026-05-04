"""
Evaluate the pipeline's performance on the hand-labeled semantic-change subset.

This is intentionally separate from eval.py (which handles mechanical metrics).
Semantic changes test pipeline behavior on subtle, judgment-requiring changes
that computed ground truth cannot cover.

Usage:
    python scripts/eval_semantic.py results/improved

Output:
    results/improved/semantic_eval.json
"""

import json
import sys
from pathlib import Path


def _load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def _find_testable_clause_ids(entry: dict) -> list[str]:
    """Determine which v1 clause IDs to test for a semantic entry.

    For merges: only the absorbed clause IDs (v1 - v2), because the
    surviving clause is tested by the mechanical eval via ChangeMap.
    For all other types: all v1_clause_ids (they equal v2_clause_ids).
    """
    v1 = set(entry["v1_clause_ids"])
    v2 = set(entry["v2_clause_ids"])
    if entry["transformation_type"] == "semantic_clause_merge":
        absorbed = v1 - v2
        return sorted(absorbed)
    return sorted(v1)


def _find_citing_wis(clause_id: str, wi_metadata: list[dict]) -> list[str]:
    """Return WI IDs that cite a given v1 clause ID."""
    return [
        wi["wi_id"]
        for wi in wi_metadata
        if any(r["clause_id"] == clause_id for r in wi["references"])
    ]


def _find_matching_proposal(
    clause_id: str, entry: dict, proposals: list[dict]
) -> dict | None:
    """Find a pipeline proposal that addresses a given semantic clause.

    Matches on:
    1. clause_reference == clause_id (direct match)
    2. clause_reference in v2_clause_ids AND old_text mentions the v1 clause
       (pipeline remapped absorbed clause to surviving one)
    """
    v2_ids = set(entry["v2_clause_ids"])
    aiqs_ref = f"AIQS {clause_id}"

    for p in proposals:
        ref = p.get("clause_reference", "")
        if ref == clause_id:
            return p
        if ref in v2_ids and aiqs_ref in p.get("old_text", ""):
            return p
    return None


_ACTION_MAP = {
    "edit_required": "edit",
    "flag_for_review": "flag",
    "no_action_required": None,
}


def _classify_outcome(
    expected_behavior: str, proposal: dict | None
) -> str:
    """Classify a single (entry, WI, clause) test point.

    Returns: 'match', 'wrong_action', or 'miss'.
    """
    expected_action = _ACTION_MAP[expected_behavior]

    if expected_action is None:
        # no_action_required: any proposal is a false positive
        return "match" if proposal is None else "wrong_action"

    if proposal is None:
        return "miss"

    actual_action = proposal.get("action", "")
    return "match" if actual_action == expected_action else "wrong_action"


def evaluate_semantic(
    results_dir: Path,
    semantic_path: Path,
    metadata_path: Path,
) -> dict:
    """Run semantic evaluation and return structured results."""
    semantic_entries = _load_json(semantic_path)
    wi_metadata = _load_json(metadata_path)

    # Pre-load all WI results
    wi_proposals: dict[str, list[dict]] = {}
    for f in sorted(results_dir.glob("WI-*.json")):
        data = _load_json(f)
        wi_proposals[f.stem] = data.get("edits", data) if isinstance(data, dict) else data

    entries_out = []
    counters = {"testable": 0, "not_testable": 0, "match": 0, "wrong_action": 0, "miss": 0}
    by_category: dict[str, dict[str, int]] = {}

    for entry in semantic_entries:
        category = entry["transformation_type"]
        if category not in by_category:
            by_category[category] = {"testable": 0, "not_testable": 0, "match": 0, "wrong_action": 0, "miss": 0}

        testable_ids = _find_testable_clause_ids(entry)
        expected = entry["expected_behavior"]

        for clause_id in testable_ids:
            citing_wis = _find_citing_wis(clause_id, wi_metadata)

            if not citing_wis:
                counters["not_testable"] += 1
                by_category[category]["not_testable"] += 1
                entries_out.append({
                    "clause_id": clause_id,
                    "category": category,
                    "expected_behavior": expected,
                    "wis_citing": [],
                    "outcome": "not_testable",
                    "detail": f"No WI cites clause {clause_id}",
                })
                continue

            # Each citing WI is an independent test point
            for wi_id in citing_wis:
                proposals = wi_proposals.get(wi_id, [])
                match = _find_matching_proposal(clause_id, entry, proposals)
                outcome = _classify_outcome(expected, match)

                counters["testable"] += 1
                counters[outcome] += 1
                by_category[category]["testable"] += 1
                by_category[category][outcome] += 1

                detail_parts = []
                if match:
                    detail_parts.append(
                        f"Proposal found: action={match.get('action')}, "
                        f"clause_reference={match.get('clause_reference')}"
                    )
                else:
                    detail_parts.append(f"No proposal found for clause {clause_id}")
                detail_parts.append(f"Expected: {expected} → {outcome}")

                entries_out.append({
                    "clause_id": clause_id,
                    "category": category,
                    "expected_behavior": expected,
                    "wi_id": wi_id,
                    "outcome": outcome,
                    "matched_proposal": {
                        "clause_reference": match.get("clause_reference"),
                        "action": match.get("action"),
                    } if match else None,
                    "detail": "; ".join(detail_parts),
                })

    return {
        "summary": counters,
        "by_category": by_category,
        "entries": entries_out,
    }


def print_summary(results: dict) -> None:
    s = results["summary"]
    print("\n=== Semantic Eval Summary ===")
    print(f"Testable instances: {s['testable']}  |  Not testable: {s['not_testable']}")
    if s["testable"] > 0:
        print(f"  Match:        {s['match']}/{s['testable']} ({100*s['match']/s['testable']:.0f}%)")
        print(f"  Wrong action: {s['wrong_action']}/{s['testable']} ({100*s['wrong_action']/s['testable']:.0f}%)")
        print(f"  Miss:         {s['miss']}/{s['testable']} ({100*s['miss']/s['testable']:.0f}%)")

    print("\n--- By Category ---")
    for cat, c in results["by_category"].items():
        label = cat.replace("semantic_", "")
        nt = f" (+{c['not_testable']} not testable)" if c["not_testable"] else ""
        if c["testable"] == 0:
            print(f"  {label:<20s}  no testable instances{nt}")
        else:
            print(
                f"  {label:<20s}  "
                f"match={c['match']}  wrong={c['wrong_action']}  miss={c['miss']}  "
                f"(n={c['testable']}){nt}"
            )

    print("\n--- Per Entry ---")
    for e in results["entries"]:
        wi = e.get("wi_id", "—")
        print(f"  {e['clause_id']}  {wi:<8s}  {e['outcome']:<14s}  {e['detail']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_semantic.py <results_dir>")
        print("  e.g. python scripts/eval_semantic.py results/improved")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    # Resolve data paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    semantic_path = project_root / "data" / "ground_truth" / "semantic_changes.json"
    metadata_path = project_root / "data" / "wi_metadata.json"

    for p in [semantic_path, metadata_path]:
        if not p.exists():
            print(f"Error: {p} not found")
            sys.exit(1)

    results = evaluate_semantic(results_dir, semantic_path, metadata_path)
    print_summary(results)

    out_path = results_dir / "semantic_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
