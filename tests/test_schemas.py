"""Schema contract tests — Pydantic accepts well-formed edits, rejects bad ones.

Cheap insurance against silent schema drift. If someone changes the
clause_reference regex or removes a required field, these tests catch it
before the eval produces nonsense.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schemas import EditProposal, ProposalAction


class TestEditProposal:
    def test_valid_proposal_accepted(self):
        """Well-formed edit proposal passes validation."""
        p = EditProposal(
            clause_reference="3.2.1",
            action=ProposalAction.EDIT,
            rationale="Clause updated in v2.",
            old_text="Calibrate per AIQS 3.2.1.",
            new_text="Verify calibration per AIQS 3.2.1.",
        )
        assert p.clause_reference == "3.2.1"
        assert p.action == ProposalAction.EDIT

    def test_prefixed_clause_reference_rejected(self):
        """Clause reference in 'AIQS 3.2.1' format must be rejected.

        Inside the schema, clause_reference is dotted-only. The prefix is
        added at rendering time.
        """
        with pytest.raises(ValidationError):
            EditProposal(
                clause_reference="AIQS 3.2.1",
                action=ProposalAction.EDIT,
                rationale="Updated.",
                old_text="Some old text.",
                new_text="Some new text.",
            )

    def test_missing_old_text_rejected(self):
        """old_text is required and must be non-empty."""
        with pytest.raises(ValidationError):
            EditProposal(
                clause_reference="3.2.1",
                action=ProposalAction.EDIT,
                rationale="Updated.",
                old_text="",
                new_text="Some new text.",
            )
