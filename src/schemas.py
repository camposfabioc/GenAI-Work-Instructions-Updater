"""Pydantic schemas for the GenAI WI Updater project.

Two groups of models live here:

1. Corpus data model — Standard / Chapter / Clause / WorkInstruction /
   TransformationLog. Used by data_gen.py to build v1, transform to v2,
   and build the WI corpus. Markdown rendering is derived from these.

2. Pipeline edit schema — EditProposal. Used by the baseline and improved
   pipelines to structure their LLM output, and by the validators to check
   reference existence, entailment, and terminology.

Keeping both groups here because they share vocabulary (clause_id format,
version tags) and the tests in tests/test_schemas.py cover both.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Corpus data model
# ---------------------------------------------------------------------------

# AIQS clause IDs look like "3.2.1" (three integers, dot-separated).
# Prefix "AIQS " is added only when the ID is cited from outside the standard
# (e.g. from WI prose). Internal to a Clause object, we store only the dotted
# form.
_CLAUSE_ID_PATTERN = r"^\d+\.\d+\.\d+$"


class Clause(BaseModel):
    """A single clause in the AIQS standard.

    `clause_id` is the dotted form (e.g. "3.2.1"). The "AIQS " prefix is added
    only at rendering time or when cited from WIs.
    """

    clause_id: str = Field(..., pattern=_CLAUSE_ID_PATTERN)
    heading: str
    body: str

    @field_validator("heading", "body")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("heading and body must be non-empty")
        return v


class Section(BaseModel):
    """A section holds an ordered list of clauses.

    Section number is the second integer in `clause_id` (e.g. clauses in
    section 3.2.x have section_number=2, chapter_number=3).
    """

    section_number: int
    title: str
    clauses: list[Clause] = Field(default_factory=list)


class Chapter(BaseModel):
    chapter_number: int
    title: str
    sections: list[Section] = Field(default_factory=list)


class Standard(BaseModel):
    """Acme Industrial Quality Standard — either v1 or v2."""

    version: Literal["v1", "v2"]
    title: str = "Acme Industrial Quality Standard"
    chapters: list[Chapter] = Field(default_factory=list)

    def all_clauses(self) -> list[Clause]:
        """Flatten every clause across chapters and sections, in document order."""
        out: list[Clause] = []
        for ch in self.chapters:
            for sec in ch.sections:
                out.extend(sec.clauses)
        return out

    def clause_by_id(self, clause_id: str) -> Clause | None:
        for cl in self.all_clauses():
            if cl.clause_id == clause_id:
                return cl
        return None


# ---------------------------------------------------------------------------
# Work Instructions
# ---------------------------------------------------------------------------


class LengthBucket(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class PositionBucket(str, Enum):
    """Section position within a WI, normalized to [0,1] then bucketed.

    Top:    [0.00, 0.33)
    Middle: [0.33, 0.67)
    Bottom: [0.67, 1.00]
    """

    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


class ClauseReference(BaseModel):
    """A single clause cited from inside a WI.

    `position_bucket` is recorded so the lost-in-middle metric can bucket
    without re-parsing the WI. It is filled at generation time from the
    section index where the reference is planted.
    """

    clause_id: str = Field(..., pattern=_CLAUSE_ID_PATTERN)
    position_bucket: PositionBucket
    is_edit_requiring: bool = False  # True if this ref hits a v1→v2 change


class WorkInstruction(BaseModel):
    """A generated Work Instruction.

    `body_markdown` is the rendered WI text (already includes section
    headings and "AIQS X.Y.Z" citations in the prose). `references` is the
    authoritative structured list the ground-truth computation uses.
    """

    wi_id: str  # e.g. "WI-014"
    title: str
    length_bucket: LengthBucket
    topic: str
    references: list[ClauseReference]
    body_markdown: str


# ---------------------------------------------------------------------------
# Transformation log
# ---------------------------------------------------------------------------


class TransformationType(str, Enum):
    RENUMBER = "renumber"
    TERM_REPLACE = "term_replace"
    STRENGTHEN = "strengthen"
    SCOPE_NARROW = "scope_narrow"
    CLAUSE_SPLIT = "clause_split"
    DEPRECATE = "deprecate"
    CLAUSE_INSERT = "clause_insert"
    # Semantic-change categories (only applied via apply_semantic_changes;
    # compute_expected_edits() filters these out).
    SEMANTIC_TONE_SHIFT = "semantic_tone_shift"
    SEMANTIC_XREF_CHAIN = "semantic_xref_chain"
    SEMANTIC_AMBIGUOUS_SCOPE = "semantic_ambiguous_scope"
    SEMANTIC_CLAUSE_MERGE = "semantic_clause_merge"


_SEMANTIC_TYPES = {
    TransformationType.SEMANTIC_TONE_SHIFT,
    TransformationType.SEMANTIC_XREF_CHAIN,
    TransformationType.SEMANTIC_AMBIGUOUS_SCOPE,
    TransformationType.SEMANTIC_CLAUSE_MERGE,
}


class TransformationEntry(BaseModel):
    """One record in the transformation log.

    `v1_clause_ids` and `v2_clause_ids` are lists because splits (1→2) and
    merges (2→1) are not one-to-one. For 1:1 transformations both lists are
    length 1. For insertions v1_clause_ids is empty; for deprecations
    v2_clause_ids is empty.
    """

    transformation_type: TransformationType
    v1_clause_ids: list[str]
    v2_clause_ids: list[str]
    # Free-form field used differently per type:
    #   - term_replace: {"from": "2847.310.0042", "to": "2847.310.0043"}
    #   - strengthen:   {"from_phrase": "should", "to_phrase": "shall"}
    #   - scope_narrow: {"from_scope": "all equipment",
    #                    "to_scope": "all Class-B rotating equipment"}
    #   - semantic_*:   {"description": "..."} — scaffold uses this
    payload: dict[str, str] = Field(default_factory=dict)

    @property
    def is_semantic(self) -> bool:
        return self.transformation_type in _SEMANTIC_TYPES


class TransformationLog(BaseModel):
    """Ordered record of every v1→v2 change."""

    entries: list[TransformationEntry] = Field(default_factory=list)

    def mechanical_entries(self) -> list[TransformationEntry]:
        return [e for e in self.entries if not e.is_semantic]

    def semantic_entries(self) -> list[TransformationEntry]:
        return [e for e in self.entries if e.is_semantic]


# ---------------------------------------------------------------------------
# Pipeline edit schema (used by pipelines.py and validators.py later)
# ---------------------------------------------------------------------------


class ProposalAction(str, Enum):
    """What the pipeline recommends for a given clause reference.

    - ``edit``: the WI text needs to be updated; ``new_text`` is the proposed
      replacement.
    - ``flag``: the WI cannot be auto-edited and a human reviewer must decide
      (e.g. the cited clause was deprecated, or the change is semantic).
      ``new_text`` is empty.
    """

    EDIT = "edit"
    FLAG = "flag"


class EditProposal(BaseModel):
    """A single proposed action for a WI, as output by the generation pipeline.

    ``clause_reference`` is the bare dotted form (e.g. "3.2.1"), matching the
    internal ``Clause.clause_id`` format. Validators add/strip the "AIQS "
    prefix as needed.

    ``new_text`` is required when ``action == EDIT`` and must be empty/blank
    when ``action == FLAG``.
    """

    clause_reference: str = Field(..., pattern=_CLAUSE_ID_PATTERN)
    action: ProposalAction
    rationale: str
    old_text: str
    new_text: str = ""

    @field_validator("rationale", "old_text")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale and old_text must be non-empty")
        return v

    @model_validator(mode="after")
    def _new_text_consistent_with_action(self) -> "EditProposal":
        if self.action == ProposalAction.EDIT and not self.new_text.strip():
            raise ValueError("new_text is required when action='edit'")
        return self


class EditProposalList(BaseModel):
    """Wrapper used when asking the LLM to return a list of edits in one call."""

    edits: list[EditProposal] = Field(default_factory=list)
