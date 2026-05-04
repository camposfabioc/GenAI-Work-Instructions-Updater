"""Unit tests for validators.py — pure functions + mocked LLM.

High-leverage: a broken validator invalidates every metric downstream.
LLM calls are mocked so tests are fast, free, and CI-friendly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import EditProposal, ProposalAction, Standard, Chapter, Section, Clause
from validators import (
    validate_reference,
    validate_entailment,
    validate_glossary,
    _is_citation_only_change,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_V2_CLAUSE_IDS = {"3.2.1", "4.2.1", "1.1.1", "2.3.4", "5.2.1"}


def _make_proposal(
    clause_ref: str = "3.2.1",
    action: ProposalAction = ProposalAction.EDIT,
    old_text: str = "Check as per AIQS 3.2.1.",
    new_text: str = "Verify as per AIQS 3.2.1.",
    rationale: str = "Updated wording.",
) -> EditProposal:
    return EditProposal(
        clause_reference=clause_ref,
        action=action,
        rationale=rationale,
        old_text=old_text,
        new_text=new_text,
    )


def _make_v2_with_clause(clause_id: str = "3.2.1", body: str = "Clause body.") -> Standard:
    return Standard(
        version="v2",
        chapters=[
            Chapter(
                chapter_number=3,
                title="Ch3",
                sections=[
                    Section(
                        section_number=2,
                        title="Sec2",
                        clauses=[Clause(clause_id=clause_id, heading="Heading", body=body)],
                    )
                ],
            )
        ],
    )


GLOSSARY = [
    {
        "term": "2847.310.0042",
        "category": "equipment_id",
        "description": "Horizontal CNC Machining Center",
        "superseded_by": "2847.310.0043",
        "common_llm_errors": ["2847-310-0042", "2847.310.42"],
    },
    {
        "term": "2847.310.0055",
        "category": "equipment_id",
        "description": "Precision Surface Grinder",
        "superseded_by": None,
        "common_llm_errors": ["2847-310-0055", "2847.310.55"],
    },
    {
        "term": "ORR",
        "category": "abbreviation",
        "description": "Operational Readiness Review",
        "superseded_by": None,
        "common_llm_errors": [],
    },
]


# ---------------------------------------------------------------------------
# validate_reference
# ---------------------------------------------------------------------------


class TestValidateReference:
    def test_existing_clause_returns_true(self):
        p = _make_proposal(clause_ref="3.2.1")
        assert validate_reference(p, _V2_CLAUSE_IDS) is True

    def test_missing_clause_returns_false(self):
        p = _make_proposal(clause_ref="9.9.9")
        assert validate_reference(p, {"3.2.1", "4.2.1"}) is False


# ---------------------------------------------------------------------------
# validate_entailment
# ---------------------------------------------------------------------------


class TestValidateEntailment:
    def test_entailed_returns_true(self):
        v2 = _make_v2_with_clause("3.2.1", "All torque equipment shall be calibrated.")
        p = _make_proposal(
            clause_ref="3.2.1",
            old_text="Calibrate tools per AIQS 3.2.1.",
            new_text="All torque equipment shall be calibrated per AIQS 3.2.1.",
        )

        mock_judgment = MagicMock()
        mock_judgment.entailed = True
        mock_judgment.rationale = "Aligns with clause."

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(parsed=mock_judgment))]

        with patch("validators.call_llm", return_value=mock_resp):
            assert validate_entailment(p, v2) is True

    def test_not_entailed_returns_false(self):
        v2 = _make_v2_with_clause("3.2.1", "Calibration is monthly.")
        p = _make_proposal(
            clause_ref="3.2.1",
            old_text="Calibrate weekly per AIQS 3.2.1.",
            new_text="Calibrate daily per AIQS 3.2.1.",
        )

        mock_judgment = MagicMock()
        mock_judgment.entailed = False
        mock_judgment.rationale = "Daily not supported by clause."

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(parsed=mock_judgment))]

        with patch("validators.call_llm", return_value=mock_resp):
            assert validate_entailment(p, v2) is False

    def test_llm_failure_returns_false(self):
        """Safety net: if the LLM call explodes, default to reject."""
        v2 = _make_v2_with_clause("3.2.1", "Some clause body.")
        p = _make_proposal(
            clause_ref="3.2.1",
            old_text="Old text here.",
            new_text="New text here.",
        )

        with patch("validators.call_llm", side_effect=Exception("API down")):
            assert validate_entailment(p, v2) is False

    def test_citation_only_change_skips_llm(self):
        """Citation-only changes return True without calling the LLM."""
        v2 = _make_v2_with_clause("3.2.1", "Some clause body.")
        p = _make_proposal(
            clause_ref="3.2.1",
            old_text="Check as per AIQS 3.4.4.",
            new_text="Check as per AIQS 3.2.1.",
        )

        with patch("validators.call_llm") as mock_llm:
            result = validate_entailment(p, v2)
            assert result is True
            mock_llm.assert_not_called()

    def test_clause_not_in_v2_returns_false(self):
        """If clause_reference doesn't resolve to a clause object, reject."""
        v2 = _make_v2_with_clause("4.2.1", "Different clause.")
        p = _make_proposal(clause_ref="3.2.1")
        assert validate_entailment(p, v2) is False


# ---------------------------------------------------------------------------
# _is_citation_only_change
# ---------------------------------------------------------------------------


class TestIsCitationOnlyChange:
    def test_same_text_different_ref(self):
        assert _is_citation_only_change(
            "as per AIQS 3.4.4", "as per AIQS 3.2.1"
        ) is True

    def test_different_content(self):
        assert _is_citation_only_change(
            "calibrate tools per AIQS 3.4.4",
            "verify tools per AIQS 3.2.1",
        ) is False


# ---------------------------------------------------------------------------
# validate_glossary
# ---------------------------------------------------------------------------


class TestValidateGlossary:
    def test_correct_migration(self):
        """Superseded ID replaced with correct successor."""
        p = _make_proposal(
            new_text="Use asset 2847.310.0043 for this procedure.",
        )
        result = validate_glossary(p, GLOSSARY)
        assert result["preservation"] is True
        assert result["migration"] is True

    def test_failed_migration(self):
        """Superseded ID still present in new_text."""
        p = _make_proposal(
            new_text="Use asset 2847.310.0042 for this procedure.",
        )
        result = validate_glossary(p, GLOSSARY)
        assert result["migration"] is False

    def test_preservation_intact(self):
        """Non-superseded ID kept in exact format."""
        p = _make_proposal(
            new_text="Grinder 2847.310.0055 must be checked.",
        )
        result = validate_glossary(p, GLOSSARY)
        assert result["preservation"] is True
        assert result["migration"] is True

    def test_preservation_mangled(self):
        """Non-superseded ID in mangled format without correct format."""
        p = _make_proposal(
            new_text="Grinder 2847-310-0055 must be checked.",
        )
        result = validate_glossary(p, GLOSSARY)
        assert result["preservation"] is False

    def test_no_ids_in_text(self):
        """No equipment IDs at all — nothing to check, passes."""
        p = _make_proposal(new_text="Verify calibration status.")
        result = validate_glossary(p, GLOSSARY)
        assert result["preservation"] is True
        assert result["migration"] is True
