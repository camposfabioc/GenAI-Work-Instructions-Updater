"""Ground-truth consistency check.

For transformation types that preserve the clause ID (strengthen,
term_replace, clause_split), the clause_id in expected_edits.json
must exist in v2. If this fails, the corpus generation has a bug
and eval metrics are unreliable.

Excluded from the check:
- ``renumber``: the v1 ID moves to a new ID in v2 by definition.
- ``deprecate``: the clause is removed from v2 by definition.
- ``scope_narrow``: may coincide with a cascading renumber.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import Standard

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _have_required_files() -> bool:
    return all(
        p.exists()
        for p in [
            _DATA_DIR / "standards" / "acme_qs_v2.json",
            _DATA_DIR / "ground_truth" / "expected_edits.json",
        ]
    )


@pytest.mark.skipif(
    not _have_required_files(),
    reason="Corpus not on disk — run data_gen.py first.",
)
class TestGroundTruthConsistency:
    def test_edit_required_clause_refs_exist_in_v2(self):
        """For types that preserve the clause ID, clause_id must exist in v2."""
        # These types apply content changes only — the ID is the same in v1 and v2
        ID_PRESERVED_TYPES = {"strengthen", "term_replace", "clause_split"}

        v2 = Standard.model_validate(
            _load_json(_DATA_DIR / "standards" / "acme_qs_v2.json")
        )
        v2_ids = {c.clause_id for c in v2.all_clauses()}

        expected = _load_json(_DATA_DIR / "ground_truth" / "expected_edits.json")

        checkable = [
            e for e in expected
            if e.get("transformation_type") in ID_PRESERVED_TYPES
        ]

        missing = []
        for entry in checkable:
            ref = entry["clause_id"]
            if ref not in v2_ids:
                missing.append(
                    f"{entry['wi_id']}: {ref} (type={entry['transformation_type']})"
                )

        assert not missing, (
            f"{len(missing)} clause_id(s) in expected_edits.json "
            f"not found in v2 (should be preserved by their transformation type):\n"
            + "\n".join(missing)
        )
