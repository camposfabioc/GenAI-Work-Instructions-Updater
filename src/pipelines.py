"""Edit-generation pipelines.

``baseline()`` — a deliberately naive pipeline that reproduces the four PoC
failure modes (reference hallucination, substantive hallucination,
lost-in-middle, rule inconsistency).

``improved()`` — per-chunk processing with citation-aware hybrid retrieval,
glossary injection, and deterministic flag logic for deprecated clauses.
"""

from __future__ import annotations

import re

from chunker import Chunk, chunk_wi
from llm import CHEAP_MODEL, call_llm
from retriever import ChangeMap, ClauseRetriever, build_change_map
from schemas import (
    Clause,
    EditProposal,
    EditProposalList,
    ProposalAction,
    Standard,
)

_BASELINE_SYSTEM_PROMPT = """You are a quality engineer reviewing Work \
Instructions against a revised quality standard.

Your task: identify text in the Work Instruction that must be edited, or \
flagged for human review, because the cited clauses in the standard have \
changed in version 2.

For each item, provide:
- clause_reference: the clause ID in DOTTED FORM ONLY (e.g. "4.2.1"). \
Do NOT include the "AIQS " prefix.
- action: either "edit" or "flag".
  * Use "edit" when you can propose a concrete replacement text.
  * Use "flag" when the cited clause was removed/deprecated in v2, or when \
the change is too ambiguous to auto-edit and a human must review.
- rationale: a brief explanation.
- old_text: the exact text from the Work Instruction that needs attention.
- new_text: the proposed replacement text. Required when action="edit". \
Leave empty ("") when action="flag".

Return only items that are strictly necessary based on changes in the \
standard. If no action is needed, return an empty list."""


def baseline(wi_markdown: str, v2_markdown: str) -> EditProposalList:
    """Naive baseline pipeline — single LLM call, no RAG, no glossary.

    Parameters
    ----------
    wi_markdown : str
        Full Markdown text of the Work Instruction.
    v2_markdown : str
        Full Markdown text of the v2 standard.

    Returns
    -------
    EditProposalList
        Structured list of proposed edits. May be empty.
    """
    user_message = (
        "Here is version 2 of the quality standard:\n\n"
        "---\n"
        f"{v2_markdown}\n"
        "---\n\n"
        "Here is the Work Instruction to review:\n\n"
        "---\n"
        f"{wi_markdown}\n"
        "---\n\n"
        "Identify all edits needed."
    )

    messages = [
        {"role": "system", "content": _BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = call_llm(
        messages=messages,
        model=CHEAP_MODEL,
        temperature=0.0,
        response_format=EditProposalList,
    )

    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Improved pipeline — Day 3
# ---------------------------------------------------------------------------

_AIQS_CITE_RE = re.compile(r"\bAIQS\s+(\d+\.\d+\.\d+)\b")

# Similarity threshold for narrowed semantic fallback.
# Below this, the citation is flagged as likely deprecated.
_FALLBACK_SIMILARITY_THRESHOLD = 0.7

_IMPROVED_SYSTEM_PROMPT = """\
You are a quality engineer updating a Work Instruction to comply with \
version 2 of the Acme Industrial Quality Standard (AIQS).

You are given:
1. A SECTION of the Work Instruction to review.
2. The RELEVANT V2 CLAUSES that the section's citations now point to.
3. A GLOSSARY of internal terminology (equipment IDs and abbreviations).
4. A CLAUSES TO EVALUATE list — the only clause IDs you must assess.

Your task: for each clause ID in CLAUSES TO EVALUATE whose content has \
changed in v2, propose an edit.

Rules:
- clause_reference: the clause ID in DOTTED FORM ONLY (e.g. "4.2.1"). \
Do NOT include the "AIQS " prefix.
- action: "edit" when you can propose concrete replacement text.
- old_text: the EXACT text from the Work Instruction that needs updating. \
Copy it verbatim — do not paraphrase.
- new_text: the proposed replacement. Must be faithful to the v2 clause \
content. Use terminology from the glossary exactly as written.
- rationale: one sentence explaining the change.
- Preserve the structure of AIQS citations in new_text. If a clause was \
renumbered, update only the clause number (e.g. "AIQS 2.3.4" → \
"AIQS 2.1.5"). Never replace an AIQS citation with a paraphrased \
description such as "as per the relevant requirements".
- Make the minimum change required. Do not rewrite sentences beyond what \
the v2 clause change strictly demands. If only the clause number changed, \
only the clause number changes in new_text.
- If a citation's v2 clause is identical to the v1 clause it replaced, \
no edit is needed — do NOT include it.
- Do NOT generate edits for clause IDs that are not in CLAUSES TO EVALUATE, \
even if other AIQS citations appear in the section text.
- Do NOT propose edits for equipment IDs (e.g. 2847.XXX.XXXX) — \
these are handled separately and must not appear in your output.
- Return an empty list if no edits are needed for this section.
- Do not append clause headings to AIQS citations in new_text. Citations must remain in the form 'AIQS X.Y.Z' only.

IMPORTANT: Do not invent requirements. Every claim in new_text must be \
directly supported by the cited v2 clause."""


def _extract_heading_hierarchy(wi_markdown: str) -> str:
    """Extract all heading lines from a WI to give the LLM document context.

    Returns a compact outline like:
        # WI-014: Setup CNC
        ## 1. Purpose
        ## 2. Scope
        ## 4. Procedure
        ### 4.1 Pre-checks
    """
    lines = []
    for line in wi_markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            lines.append(stripped)
    return "\n".join(lines)


def _format_glossary_block(glossary: list[dict]) -> str:
    """Format glossary entries for prompt injection."""
    parts = []
    for entry in glossary:
        term = entry["term"]
        desc = entry["description"]
        cat = entry.get("category", "")
        superseded = entry.get("superseded_by")
        line = f"- {term}: {desc} [{cat}]"
        if superseded:
            line += f" → SUPERSEDED BY {superseded} in v2"
        parts.append(line)
    return "\n".join(parts)


def _format_clauses_block(clauses: list[Clause]) -> str:
    """Format retrieved v2 clauses for the prompt context."""
    parts = []
    for clause in clauses:
        parts.append(
            f"### AIQS {clause.clause_id} — {clause.heading}\n{clause.body}"
        )
    return "\n\n".join(parts)


def _scan_terminology(
    chunk_text: str,
    glossary: list[dict],
) -> list[EditProposal]:
    """Scan a chunk for superseded equipment IDs and generate edits directly.

    Used for chunks with no AIQS citations — no LLM call needed.
    """
    edits: list[EditProposal] = []
    for entry in glossary:
        if entry.get("category") != "equipment_id":
            continue
        old_id = entry["term"]
        new_id = entry.get("superseded_by")
        if not new_id or old_id not in chunk_text:
            continue
        # Find a sentence-level context around the old ID
        for line in chunk_text.splitlines():
            if old_id in line:
                edits.append(
                    EditProposal(
                        clause_reference="0.0.0",  # placeholder — no clause cited
                        action=ProposalAction.EDIT,
                        rationale=(
                            f"Equipment ID {old_id} has been superseded by "
                            f"{new_id} in v2."
                        ),
                        old_text=line.strip(),
                        new_text=line.strip().replace(old_id, new_id),
                    )
                )
                break  # one edit per superseded ID per chunk
    return edits


def _process_chunk(
    chunk: Chunk,
    heading_hierarchy: str,
    v1: Standard,
    retriever: ClauseRetriever,
    change_map: ChangeMap,
    glossary: list[dict],
    glossary_block: str,
) -> list[EditProposal]:
    """Process a single WI chunk through the improved pipeline.

    Returns a list of EditProposals (may be empty).
    """
    # Step 1: extract AIQS citations from this chunk
    citations = _AIQS_CITE_RE.findall(chunk.text)

    if not citations:
        # No citations — skip LLM, but scan for superseded terminology
        return _scan_terminology(chunk.text, glossary)

    # Step 2: resolve each citation via change_map
    retrieved_clauses: list[Clause] = []
    flag_proposals: list[EditProposal] = []
    seen_clause_ids: set[str] = set()

    for v1_id in citations:
        if v1_id in seen_clause_ids:
            continue
        seen_clause_ids.add(v1_id)

        if v1_id in change_map.unchanged:
            # Clause exists in v2 with identical body — no edit needed, skip
            continue

        elif v1_id in change_map.changed:
            # Clause exists in v2 but body differs — send to LLM
            clause = retriever.retrieve_by_id(v1_id)
            if clause:
                retrieved_clauses.append(clause)

        elif v1_id in change_map.id_collision:
            # A different clause now occupies this ID in v2 (cascading
            # renumber).  Search ALL of v2 for where the original clause
            # went — the target could be in v2_only, changed, or unchanged.
            v1_clause = v1.clause_by_id(v1_id)
            if v1_clause:
                query = f"{v1_clause.clause_id} {v1_clause.heading}\n{v1_clause.body}"
                matches = retriever.retrieve(query, k=3)
                if matches and matches[0][1] >= _FALLBACK_SIMILARITY_THRESHOLD:
                    retrieved_clauses.append(matches[0][0])
                else:
                    old_text = ""
                    for line in chunk.text.splitlines():
                        if f"AIQS {v1_id}" in line:
                            old_text = line.strip()
                            break
                    if not old_text:
                        old_text = f"Reference to AIQS {v1_id}"
                    flag_proposals.append(
                        EditProposal(
                            clause_reference=v1_id,
                            action=ProposalAction.FLAG,
                            rationale=(
                                f"AIQS {v1_id} in v2 contains unrelated "
                                f"content (likely renumber cascade) and no "
                                f"similar clause was found elsewhere — "
                                f"human review required."
                            ),
                            old_text=old_text,
                            new_text="",
                        )
                    )

        elif v1_id in change_map.v1_only:
            # Citation points to a clause not in v2.
            # Try narrowed semantic search among v2-only clauses.
            v1_clause = v1.clause_by_id(v1_id)
            if v1_clause:
                query = f"{v1_clause.clause_id} {v1_clause.heading}\n{v1_clause.body}"
                matches = retriever.retrieve_narrowed(
                    query, change_map.v2_only, k=3
                )
                if matches and matches[0][1] >= _FALLBACK_SIMILARITY_THRESHOLD:
                    # Best match is above threshold — likely renumber/split
                    retrieved_clauses.append(matches[0][0])
                else:
                    # No good match — flag as likely deprecated
                    old_text = ""
                    for line in chunk.text.splitlines():
                        if f"AIQS {v1_id}" in line:
                            old_text = line.strip()
                            break
                    if not old_text:
                        old_text = f"Reference to AIQS {v1_id}"

                    flag_proposals.append(
                        EditProposal(
                            clause_reference=v1_id,
                            action=ProposalAction.FLAG,
                            rationale=(
                                f"AIQS {v1_id} was not found in v2 and no "
                                f"similar replacement clause was identified. "
                                f"The clause may have been deprecated — "
                                f"human review required."
                            ),
                            old_text=old_text,
                            new_text="",
                        )
                    )
            else:
                # v1 clause not found in v1 standard either (shouldn't happen)
                flag_proposals.append(
                    EditProposal(
                        clause_reference=v1_id,
                        action=ProposalAction.FLAG,
                        rationale=(
                            f"AIQS {v1_id} not found in v1 or v2. "
                            f"Human review required."
                        ),
                        old_text=f"Reference to AIQS {v1_id}",
                        new_text="",
                    )
                )
        # else: not in v1 or v2 sets — shouldn't happen with well-formed data

    # Step 3: if we have retrieved clauses, make one LLM call for this chunk
    llm_proposals: list[EditProposal] = []
    if retrieved_clauses:
        clauses_block = _format_clauses_block(retrieved_clauses)
        evaluated_ids = ", ".join(c.clause_id for c in retrieved_clauses)

        user_message = (
            "WORK INSTRUCTION OUTLINE:\n"
            f"{heading_hierarchy}\n\n"
            "---\n\n"
            "SECTION TO REVIEW:\n"
            f"{chunk.text}\n\n"
            "---\n\n"
            "RELEVANT V2 CLAUSES:\n"
            f"{clauses_block}\n\n"
            "---\n\n"
            "GLOSSARY:\n"
            f"{glossary_block}\n\n"
            "---\n\n"
            "CLAUSES TO EVALUATE:\n"
            f"{evaluated_ids}\n\n"
            "---\n\n"
            "Identify edits needed for this section, restricted to the "
            "clause IDs listed in CLAUSES TO EVALUATE."
        )

        messages = [
            {"role": "system", "content": _IMPROVED_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = call_llm(
            messages=messages,
            model=CHEAP_MODEL,
            temperature=0.0,
            response_format=EditProposalList,
        )
        result = response.choices[0].message.parsed
        if result:
            # Filter degenerate proposals where no change was actually made
            llm_proposals = [
                p for p in result.edits
                if p.old_text.strip() != p.new_text.strip()
            ]

    # Also scan for superseded terminology even in chunks with citations
    term_proposals = _scan_terminology(chunk.text, glossary)

    # Deduplicate: if an LLM proposal already covers a superseded ID, skip
    llm_old_texts = {p.old_text for p in llm_proposals}
    term_proposals = [
        p for p in term_proposals if p.old_text not in llm_old_texts
    ]

    return flag_proposals + llm_proposals + term_proposals


def improved(
    wi_markdown: str,
    v1: Standard,
    v2: Standard,
    glossary: list[dict],
    retriever: ClauseRetriever,
    change_map: ChangeMap,
) -> EditProposalList:
    """Improved pipeline — per-chunk RAG with hybrid retrieval and glossary.

    Parameters
    ----------
    wi_markdown : str
        Full Markdown text of the Work Instruction.
    v1 : Standard
        The v1 standard (for looking up v1 clause bodies during fallback).
    v2 : Standard
        The v2 standard (indexed in the retriever).
    glossary : list[dict]
        Glossary entries for terminology injection.
    retriever : ClauseRetriever
        Pre-built retriever over v2 clauses.
    change_map : ChangeMap
        Structural diff between v1 and v2 clause IDs.

    Returns
    -------
    EditProposalList
        Aggregated proposals across all chunks.
    """
    chunks = chunk_wi(wi_markdown)
    heading_hierarchy = _extract_heading_hierarchy(wi_markdown)
    glossary_block = _format_glossary_block(glossary)

    all_edits: list[EditProposal] = []

    for chunk in chunks:
        chunk_edits = _process_chunk(
            chunk=chunk,
            heading_hierarchy=heading_hierarchy,
            v1=v1,
            retriever=retriever,
            change_map=change_map,
            glossary=glossary,
            glossary_block=glossary_block,
        )
        all_edits.extend(chunk_edits)

    return EditProposalList(edits=all_edits)
