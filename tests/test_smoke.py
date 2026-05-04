"""Smoke test — validate_all() runs on 2 WIs without exploding.

Does NOT check correctness of values — only that the wiring works:
imports resolve, files parse, types are compatible, output has the
expected structure. Catches refactoring breakage before you try to
run the full pipeline.

LLM calls are mocked to keep the test fast and free.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import Standard
from validators import validate_all

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_RESULTS_DIR = _REPO_ROOT / "results" / "improved"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _have_required_files() -> bool:
    """Check that the corpus and at least 2 WI results exist."""
    required = [
        _DATA_DIR / "standards" / "acme_qs_v2.json",
        _DATA_DIR / "glossary.json",
        _RESULTS_DIR / "WI-001.json",
        _RESULTS_DIR / "WI-002.json",
    ]
    return all(p.exists() for p in required)


@pytest.mark.skipif(
    not _have_required_files(),
    reason="Corpus or pipeline results not on disk — run data_gen.py and run_pipeline.py first.",
)
class TestSmoke:
    def test_validate_all_runs_on_two_wis(self, tmp_path):
        """validate_all() processes 2 WIs and produces well-structured output."""
        # Copy only WI-001 and WI-002 to a temp dir so we don't process all 30
        for wi in ("WI-001.json", "WI-002.json"):
            src = _RESULTS_DIR / wi
            (tmp_path / wi).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        v2 = Standard.model_validate(
            _load_json(_DATA_DIR / "standards" / "acme_qs_v2.json")
        )
        glossary = _load_json(_DATA_DIR / "glossary.json")

        # Mock entailment LLM calls
        mock_judgment = MagicMock()
        mock_judgment.entailed = True
        mock_judgment.rationale = "Mock: entailed."

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(parsed=mock_judgment))]

        with patch("validators.call_llm", return_value=mock_resp):
            results = validate_all(tmp_path, v2, glossary)

        # Structure checks
        assert isinstance(results, list)
        assert len(results) > 0

        for entry in results:
            assert "wi_id" in entry
            assert "clause_reference" in entry
            assert "action" in entry
            assert "ref_valid" in entry
            assert "entailment_valid" in entry
            assert "glossary_check" in entry

            # Skipped entries have None values
            if entry["action"] == "flag" or entry["clause_reference"] == "0.0.0":
                assert entry["ref_valid"] is None
                assert entry["entailment_valid"] is None
                assert entry["glossary_check"] is None
            else:
                assert isinstance(entry["ref_valid"], bool)
                assert isinstance(entry["entailment_valid"], bool)
                assert isinstance(entry["glossary_check"], dict)
                assert "preservation" in entry["glossary_check"]
                assert "migration" in entry["glossary_check"]

        # Check both WIs are represented
        wi_ids = {e["wi_id"] for e in results}
        assert "WI-001" in wi_ids
        assert "WI-002" in wi_ids
