"""Generate the full synthetic corpus for the GenAI WI Updater project.

Produces under data/:
    standards/acme_qs_v1.md
    standards/acme_qs_v2.md
    work_instructions/WI-001.md … WI-030.md
    glossary.json
    ground_truth/expected_edits.json
    ground_truth/semantic_changes_scaffold.json   (template for hand-labeling)
    transformation_log.json                        (audit trail)

See data_gen_decisions.md for the authoritative design rationale behind
every choice in this file.

Run:
    python src/data_gen.py
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI

from schemas import (
    Chapter,
    Clause,
    ClauseReference,
    EditProposal,
    EditProposalList,
    LengthBucket,
    PositionBucket,
    Section,
    Standard,
    TransformationEntry,
    TransformationLog,
    TransformationType,
    WorkInstruction,
)

# ---------------------------------------------------------------------------
# Environment & client
# ---------------------------------------------------------------------------

load_dotenv()

# Cheap tier for bulk generation per project plan §9.
MODEL_CHEAP = "gpt-4o-mini"

# Single pinned RNG seed — every random choice in this file threads from it.
# API-side non-determinism still leaks in (temperature=0 is not a strict
# guarantee); that limitation is documented in the README.
SEED = 20260423

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy OpenAI client. Fails loudly if OPENAI_API_KEY is not set."""
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to .env (see .env.example)."
            )
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
STANDARDS_DIR = DATA_DIR / "standards"
WI_DIR = DATA_DIR / "work_instructions"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"


# ---------------------------------------------------------------------------
# Standard structure — 5 chapters × 5 sections × 4 clauses = 100 clauses v1
# ---------------------------------------------------------------------------

# Chapter and section titles. These are the skeleton the LLM fills clause
# bodies into. Curated (not LLM-generated) for consistency across corpus
# regenerations.
CHAPTER_STRUCTURE: list[dict] = [
    {
        "chapter_number": 1,
        "title": "General Provisions",
        "sections": [
            {"n": 1, "title": "Scope and Applicability"},
            {"n": 2, "title": "Normative References"},
            {"n": 3, "title": "Definitions and Abbreviations"},
            {"n": 4, "title": "Roles and Responsibilities"},
            {"n": 5, "title": "Documentation Requirements"},
        ],
    },
    {
        "chapter_number": 2,
        "title": "Equipment Qualification",
        "sections": [
            {"n": 1, "title": "Asset Identification and Tagging"},
            {"n": 2, "title": "Installation Qualification"},
            {"n": 3, "title": "Operational Readiness Review"},
            {"n": 4, "title": "Periodic Requalification"},
            {"n": 5, "title": "Decommissioning"},
        ],
    },
    {
        "chapter_number": 3,
        "title": "Operational Controls",
        "sections": [
            {"n": 1, "title": "Pre-operation Checks"},
            {"n": 2, "title": "Torque and Fastener Controls"},
            {"n": 3, "title": "Tooling and Fixture Controls"},
            {"n": 4, "title": "Process Parameter Controls"},
            {"n": 5, "title": "Energy Isolation and LOTO"},
        ],
    },
    {
        "chapter_number": 4,
        "title": "Inspection and Measurement",
        "sections": [
            {"n": 1, "title": "First Article Inspection"},
            {"n": 2, "title": "In-process Inspection"},
            {"n": 3, "title": "Final Inspection and Release"},
            {"n": 4, "title": "Measurement System Analysis"},
            {"n": 5, "title": "Calibration Management"},
        ],
    },
    {
        "chapter_number": 5,
        "title": "Nonconformance and Corrective Action",
        "sections": [
            {"n": 1, "title": "Detection and Reporting"},
            {"n": 2, "title": "Containment"},
            {"n": 3, "title": "Root Cause Analysis"},
            {"n": 4, "title": "Corrective and Preventive Action"},
            {"n": 5, "title": "Records Retention"},
        ],
    },
]

CLAUSES_PER_SECTION = 4  # 5 × 5 × 4 = 100


# ---------------------------------------------------------------------------
# Internal vocabulary (glossary) — see decisions doc §6
# ---------------------------------------------------------------------------
#
# 6 equipment IDs + 4 abbreviations = 10 terms. Equipment IDs follow
# pattern {plant}.{area}.{asset}. Plant 2847 shared by all (single-plant
# fiction). Area codes: 310=machining, 420=forming, 510=metrology.
#
# Two IDs have their last digit changed v1→v2, consuming the 10
# term-replacement slots from the transformation budget (§4 of decisions doc).

GLOSSARY_ENTRIES: list[dict] = [
    # --- Equipment IDs ---
    {
        "term": "2847.310.0042",
        "category": "equipment_id",
        "description": "Horizontal CNC Machining Center",
        "introduced_in": "v1",
        "superseded_by": "2847.310.0043",  # ← v1→v2 replacement
        "common_llm_errors": [
            "2847.310.42",
            "28473100042",
            "2847-310-0042",
            "the CNC machine",
            "the machining center",
            "the horizontal CNC",
        ],
    },
    {
        "term": "2847.310.0055",
        "category": "equipment_id",
        "description": "Precision Surface Grinder",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "2847.310.55",
            "28473100055",
            "2847-310-0055",
            "the surface grinder",
            "the grinder",
        ],
    },
    {
        "term": "2847.310.0061",
        "category": "equipment_id",
        "description": "Vertical Turning Lathe",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "2847.310.61",
            "28473100061",
            "2847-310-0061",
            "the turning lathe",
            "the lathe",
            "the vertical lathe",
        ],
    },
    {
        "term": "2847.420.0018",
        "category": "equipment_id",
        "description": "Hydraulic Press, 12-ton",
        "introduced_in": "v1",
        "superseded_by": "2847.420.0019",  # ← v1→v2 replacement
        "common_llm_errors": [
            "2847.420.18",
            "28474200018",
            "2847-420-0018",
            "the hydraulic press",
            "the 12-ton press",
            "the press",
        ],
    },
    {
        "term": "2847.420.0023",
        "category": "equipment_id",
        "description": "Injection Molding Unit",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "2847.420.23",
            "28474200023",
            "2847-420-0023",
            "the injection molding unit",
            "the molding unit",
            "the injection molder",
        ],
    },
    {
        "term": "2847.510.0007",
        "category": "equipment_id",
        "description": "Coordinate Measuring Machine",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "2847.510.7",
            "28475100007",
            "2847-510-0007",
            "the CMM",
            "the coordinate measuring machine",
            "the measuring machine",
        ],
    },
    # --- Abbreviations ---
    {
        "term": "ORR",
        "category": "abbreviation",
        "description": "Operational Readiness Review",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "Operations Risk Review",
            "Operational Review",
            "Operations Readiness Report",
            "Operational Risk Review",
            "Operating Readiness Review",
        ],
    },
    {
        "term": "NCR",
        "category": "abbreviation",
        "description": "Nonconformance Report",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "Non-conformance Record",
            "Non-Conformity Report",
            "Nonconformity Report",
            "Non Conformance Report",
        ],
    },
    {
        "term": "FAI",
        "category": "abbreviation",
        "description": "First Article Inspection",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "First Article Inspection Report",
            "First Piece Inspection",
            "Initial Article Inspection",
            "First Articles Inspection",
        ],
    },
    {
        "term": "LOTO",
        "category": "abbreviation",
        "description": "Lockout-Tagout",
        "introduced_in": "v1",
        "superseded_by": None,
        "common_llm_errors": [
            "Lock Out Tag Out",
            "Lock-Out/Tag-Out",
            "LO/TO",
            "Lockout and Tagout",
            "Lockout Tagout Procedure",
        ],
    },
]


def build_glossary() -> list[dict]:
    """Return the internal-vocabulary glossary as a list of dicts.

    Pure function — returns the static GLOSSARY_ENTRIES. Kept as a function
    (not a bare module-level export) so it can be swapped for a loader in
    tests or if the glossary is ever externalized to a YAML file.
    """
    return [dict(entry) for entry in GLOSSARY_ENTRIES]


def glossary_terms_stable_v1_to_v2(glossary: list[dict]) -> list[dict]:
    """Subset of the glossary that does NOT change between v1 and v2.

    Used by the terminology preservation metric denominator.
    """
    return [e for e in glossary if e["superseded_by"] is None]


def glossary_terms_replaced_v1_to_v2(glossary: list[dict]) -> list[dict]:
    """Subset of the glossary whose term is replaced v1→v2.

    Used to drive the term-replacement transformations in transform_to_v2()
    and as the migration-rate metric denominator.
    """
    return [e for e in glossary if e["superseded_by"] is not None]


# ---------------------------------------------------------------------------
# Vocabulary placement — which glossary terms are allowed in which section
# ---------------------------------------------------------------------------
#
# Maps section IDs ("chapter.section") to the glossary terms that can appear
# in clauses of that section. Curated by hand: equipment IDs only appear in
# sections where those machines are actually used; abbreviations appear in
# sections where the concept is relevant.
#
# The RNG then picks which specific clauses within each section receive
# each allowed term, subject to the per-term frequency targets:
#   equipment IDs: ~5 clauses each
#   abbreviations: ~9 clauses each
# → ~66 total vocabulary instances across 100 clauses.

SECTION_VOCAB_ALLOWLIST: dict[str, list[str]] = {
    # --- Chapter 1: General Provisions ---
    "1.1": [],                                                       # Scope — abstract, no specific equipment
    "1.2": [],                                                       # Normative References — standards only
    "1.3": ["ORR", "NCR", "FAI", "LOTO"],                            # Definitions — naturally lists abbreviations
    "1.4": ["ORR", "NCR", "FAI", "LOTO"],                            # Roles and Responsibilities
    "1.5": ["NCR", "FAI"],                                           # Documentation Requirements
    # --- Chapter 2: Equipment Qualification ---
    "2.1": ["2847.310.0042", "2847.310.0055", "2847.310.0061",
            "2847.420.0018", "2847.420.0023", "2847.510.0007"],      # Asset Identification — every ID fits
    "2.2": ["2847.310.0042", "2847.420.0018", "2847.510.0007"],      # Installation Qualification
    "2.3": ["ORR", "2847.310.0042", "2847.420.0018"],                # ORR section — by name
    "2.4": ["2847.310.0055", "2847.310.0061", "2847.510.0007"],      # Periodic Requalification
    "2.5": ["2847.420.0023", "LOTO"],                                # Decommissioning
    # --- Chapter 3: Operational Controls ---
    "3.1": ["2847.310.0042", "2847.310.0061", "2847.420.0018"],      # Pre-operation Checks
    "3.2": ["2847.310.0042", "2847.420.0018"],                       # Torque & Fastener — CNC + press
    "3.3": ["2847.310.0055", "2847.310.0061"],                       # Tooling — grinder + lathe
    "3.4": ["2847.310.0042", "2847.420.0023"],                       # Process Parameters — CNC + injection
    "3.5": ["LOTO", "2847.420.0018", "2847.420.0023"],               # Energy Isolation & LOTO
    # --- Chapter 4: Inspection and Measurement ---
    "4.1": ["FAI", "2847.510.0007"],                                 # First Article Inspection
    "4.2": ["2847.510.0007", "2847.310.0055"],                       # In-process Inspection
    "4.3": ["FAI", "NCR"],                                           # Final Inspection & Release
    "4.4": ["2847.510.0007"],                                        # Measurement System Analysis
    "4.5": ["2847.510.0007"],                                        # Calibration Management
    # --- Chapter 5: Nonconformance and Corrective Action ---
    "5.1": ["NCR"],                                                  # Detection and Reporting
    "5.2": ["NCR", "LOTO"],                                          # Containment
    "5.3": ["NCR"],                                                  # Root Cause Analysis
    "5.4": ["NCR", "ORR"],                                           # Corrective and Preventive Action
    "5.5": ["NCR", "FAI"],                                           # Records Retention
}


def section_id(chapter_number: int, section_number: int) -> str:
    """Return the 'C.S' key used in SECTION_VOCAB_ALLOWLIST."""
    return f"{chapter_number}.{section_number}"


# ---------------------------------------------------------------------------
# Vocabulary assignment — decide which clauses get which terms
# ---------------------------------------------------------------------------


def _term_frequency_target(entry: dict) -> int:
    """How many clauses should contain this term across the whole corpus.

    Tighter for equipment IDs (~5) to keep them localized; looser for
    abbreviations (~9) because they appear naturally in more contexts.
    """
    if entry["category"] == "equipment_id":
        return 5
    return 9  # abbreviation


def assign_vocabulary(
    glossary: list[dict],
    rng: random.Random,
) -> dict[str, list[str]]:
    """Decide which terms appear in which clauses, respecting allowlist + targets.

    Returns a dict mapping clause_id ("C.S.N") → list of terms to include.
    A clause may receive 0, 1, or 2 terms. Clauses never get 3+ terms to
    avoid unrealistic density.

    Algorithm:
      1. For each glossary term, pick N candidate clauses from allowed
         sections (N = per-term frequency target). All candidate clauses
         within an allowed section are eligible.
      2. If a clause ends up with >2 terms assigned, drop the lowest-priority
         one (equipment IDs keep priority over abbreviations, since they're
         scarcer).
    """
    # Flatten: every clause id that exists in v1
    all_clause_ids: list[str] = []
    for ch in CHAPTER_STRUCTURE:
        for sec in ch["sections"]:
            sec_key = section_id(ch["chapter_number"], sec["n"])
            for clause_n in range(1, CLAUSES_PER_SECTION + 1):
                all_clause_ids.append(f"{sec_key}.{clause_n}")

    assignments: dict[str, list[str]] = {cid: [] for cid in all_clause_ids}

    # Deterministic order: sort by category (equipment_id first for priority),
    # then by term. Equipment IDs get first pick → more likely to land in
    # their small eligible pool.
    ordered_terms = sorted(
        glossary,
        key=lambda e: (0 if e["category"] == "equipment_id" else 1, e["term"]),
    )

    for entry in ordered_terms:
        term = entry["term"]
        target = _term_frequency_target(entry)
        # Eligible clause IDs: those in sections whose allowlist contains the term
        eligible: list[str] = []
        for cid in all_clause_ids:
            chap, sec, _ = cid.split(".")
            sec_key = f"{chap}.{sec}"
            if term in SECTION_VOCAB_ALLOWLIST.get(sec_key, []):
                eligible.append(cid)
        # Sample without replacement. If fewer eligible than target, take all.
        chosen = rng.sample(eligible, k=min(target, len(eligible)))
        for cid in chosen:
            assignments[cid].append(term)

    # Cap at 2 terms per clause, dropping abbreviations first (lower priority)
    for cid, terms in assignments.items():
        if len(terms) <= 2:
            continue
        # Partition: equipment_ids first, then abbreviations
        def _prio(t: str) -> int:
            for e in glossary:
                if e["term"] == t:
                    return 0 if e["category"] == "equipment_id" else 1
            return 2
        terms.sort(key=_prio)
        assignments[cid] = terms[:2]

    return assignments


# ---------------------------------------------------------------------------
# build_v1() — generate clause bodies one section at a time
# ---------------------------------------------------------------------------

# Temperature for prose generation. Decision doc §8 pins temperature=0 for
# strict reproducibility, but at 0 the model tends to repeat syntactic
# structures across clauses in the same call. 0.3 keeps the output varied
# without drifting off-spec.
PROSE_TEMPERATURE = 0.3

# Retry budget for schema-invalid LLM output. A section generation that
# fails 3 times kills the script — better to fail loudly than to produce a
# silently-incomplete corpus.
MAX_SECTION_RETRIES = 3

# Target length range per clause body. ~80 words matches the mean of real
# manufacturing-standard clauses (sampled from ISO 9001).
CLAUSE_MIN_WORDS = 45
CLAUSE_MAX_WORDS = 120


V1_SYSTEM_PROMPT = """You are drafting clauses for the Acme Industrial Quality Standard (AIQS), \
a formal manufacturing quality standard. Your clauses must read like real \
industry-standard prose: formal, third-person, declarative, and procedurally precise.

Style rules (strict):
- Use "shall" for mandatory actions, "should" for recommendations, "may" for \
permissions. Mix these naturally — not every clause is mandatory.
- Prefer passive or impersonal constructions ("records shall be retained", \
"operators are required to") over first or second person.
- No marketing language, no adjectives like "robust" or "state-of-the-art", no \
filler phrases like "it is important to note".
- Each clause body: {min_words}–{max_words} words.
- Each clause has a short noun-phrase heading (3–6 words), sentence case.
- The four clauses you generate for a section must cover DISTINCT aspects of \
the section's topic. No two clauses should restate the same point.

Terminology rules (strict):
- When a clause is listed with "required terms", those exact strings MUST appear \
verbatim in the clause body. Do not expand abbreviations, do not reformat \
equipment IDs, do not substitute synonyms.
- Equipment IDs follow the pattern NNNN.NNN.NNNN (e.g. 2847.310.0042). They are \
proprietary Acme asset tags — treat them as opaque tokens.
- Clauses without required terms should avoid inventing equipment IDs or \
abbreviations; stay at the generic procedural level.

Output format: JSON object with a single key "clauses", holding exactly four \
objects in order of the clause IDs listed. Each object has fields \
"clause_id", "heading", "body". No extra commentary outside the JSON."""


def _build_v1_user_prompt(
    chapter_number: int,
    chapter_title: str,
    section_number: int,
    section_title: str,
    clause_specs: list[tuple[str, list[str]]],
) -> str:
    """Construct the per-section user prompt from structured inputs."""
    lines: list[str] = []
    lines.append(f"Chapter {chapter_number}: {chapter_title}")
    lines.append(
        f"Section {chapter_number}.{section_number}: {section_title}"
    )
    lines.append("")
    lines.append("Generate four clauses for this section with the following IDs and terminology requirements:")
    lines.append("")
    for clause_id, required_terms in clause_specs:
        if required_terms:
            term_list = ", ".join(f'"{t}"' for t in required_terms)
            lines.append(f"- {clause_id}: required terms → {term_list}")
        else:
            lines.append(f"- {clause_id}: no required terms")
    lines.append("")
    lines.append(
        "Cover four distinct aspects of the section topic. Integrate required "
        "terms naturally — the clause should be ABOUT a topic where those terms "
        "belong, not merely name-drop them."
    )
    return "\n".join(lines)


def _parse_section_response(
    raw_json: str,
    expected_clause_ids: list[str],
) -> list[Clause]:
    """Parse and validate an LLM response for one section.

    Raises ValueError if the response is malformed, missing clauses, has
    wrong clause IDs, or has out-of-range body lengths.
    """
    import json

    data = json.loads(raw_json)
    if "clauses" not in data or not isinstance(data["clauses"], list):
        raise ValueError("response missing 'clauses' list")

    if len(data["clauses"]) != len(expected_clause_ids):
        raise ValueError(
            f"expected {len(expected_clause_ids)} clauses, "
            f"got {len(data['clauses'])}"
        )

    out: list[Clause] = []
    for expected_id, obj in zip(expected_clause_ids, data["clauses"]):
        if obj.get("clause_id") != expected_id:
            raise ValueError(
                f"expected clause_id {expected_id}, got {obj.get('clause_id')}"
            )
        body = obj.get("body", "").strip()
        word_count = len(body.split())
        if not (CLAUSE_MIN_WORDS <= word_count <= CLAUSE_MAX_WORDS):
            raise ValueError(
                f"clause {expected_id} body length {word_count} words "
                f"outside [{CLAUSE_MIN_WORDS}, {CLAUSE_MAX_WORDS}]"
            )
        out.append(Clause.model_validate(obj))
    return out


def _check_required_terms_present(
    clauses: list[Clause],
    clause_specs: list[tuple[str, list[str]]],
) -> list[str]:
    """Return a list of error messages for missing required terms. Empty if all ok."""
    errors: list[str] = []
    specs_by_id = {cid: terms for cid, terms in clause_specs}
    for cl in clauses:
        for term in specs_by_id.get(cl.clause_id, []):
            if term not in cl.body:
                errors.append(
                    f"clause {cl.clause_id} missing required term '{term}'"
                )
    return errors


def _generate_section(
    chapter_number: int,
    chapter_title: str,
    section_number: int,
    section_title: str,
    clause_specs: list[tuple[str, list[str]]],
) -> list[Clause]:
    """One section generation with retry on schema/terminology failure."""
    client = get_client()
    system_prompt = V1_SYSTEM_PROMPT.format(
        min_words=CLAUSE_MIN_WORDS, max_words=CLAUSE_MAX_WORDS
    )
    user_prompt = _build_v1_user_prompt(
        chapter_number, chapter_title, section_number, section_title, clause_specs
    )
    expected_ids = [cid for cid, _ in clause_specs]

    last_error: str = ""
    for attempt in range(1, MAX_SECTION_RETRIES + 1):
        # On retry, append the prior error so the model can self-correct.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if attempt > 1 and last_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous response failed validation: {last_error}\n"
                        "Regenerate the clauses correcting this issue. Keep the "
                        "same JSON schema and the same clause IDs."
                    ),
                }
            )

        response = client.chat.completions.create(
            model=MODEL_CHEAP,
            messages=messages,
            temperature=PROSE_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""

        try:
            clauses = _parse_section_response(raw, expected_ids)
        except (ValueError, Exception) as e:
            last_error = str(e)
            continue

        term_errors = _check_required_terms_present(clauses, clause_specs)
        if term_errors:
            last_error = "; ".join(term_errors)
            continue

        return clauses

    raise RuntimeError(
        f"Section {chapter_number}.{section_number} generation failed after "
        f"{MAX_SECTION_RETRIES} attempts. Last error: {last_error}"
    )


def build_v1(
    glossary: list[dict],
    rng: random.Random,
    verbose: bool = True,
) -> Standard:
    """Build the v1 Standard by calling the LLM once per section.

    25 LLM calls total. Each call generates 4 clauses for one section,
    with required terms injected per the vocabulary assignment.
    """
    assignments = assign_vocabulary(glossary, rng)

    std = Standard(version="v1", chapters=[])

    for ch_struct in CHAPTER_STRUCTURE:
        chapter = Chapter(
            chapter_number=ch_struct["chapter_number"],
            title=ch_struct["title"],
            sections=[],
        )
        for sec_struct in ch_struct["sections"]:
            sec_key = section_id(ch_struct["chapter_number"], sec_struct["n"])

            # Build the clause specs for this section: (clause_id, required_terms)
            clause_specs: list[tuple[str, list[str]]] = []
            for clause_n in range(1, CLAUSES_PER_SECTION + 1):
                clause_id = f"{sec_key}.{clause_n}"
                clause_specs.append((clause_id, assignments.get(clause_id, [])))

            if verbose:
                n_with_terms = sum(1 for _, t in clause_specs if t)
                print(
                    f"  generating section {sec_key} "
                    f"({sec_struct['title']}) — "
                    f"{n_with_terms}/4 clauses have required terms"
                )

            clauses = _generate_section(
                chapter_number=ch_struct["chapter_number"],
                chapter_title=ch_struct["title"],
                section_number=sec_struct["n"],
                section_title=sec_struct["title"],
                clause_specs=clause_specs,
            )

            section = Section(
                section_number=sec_struct["n"],
                title=sec_struct["title"],
                clauses=clauses,
            )
            chapter.sections.append(section)

        std.chapters.append(chapter)

    return std


# ---------------------------------------------------------------------------
# transform_to_v2() — deterministic mechanical transformations
# ---------------------------------------------------------------------------
#
# Transformation budget (locked in decisions doc §4):
#   renumber        : 8 clauses
#   term_replace    : 10 clauses (2 equipment IDs × ~5 clauses each)
#   strengthen      : 8 clauses
#   scope_narrow    : 5 clauses
#   clause_split    : 3 clauses → 6 v2 clauses
#   deprecate       : 4 clauses (removed from v2 entirely)
#   clause_insert   : 5 new v2-only clauses
#
# Order (locked):
#   1. deprecate            (structural — removes clauses)
#   2. clause_split         (structural — 1→2 clauses)
#   3. clause_insert        (structural — adds new clauses)
#   4. content passes on remaining unchanged pool:
#        term_replace → strengthen → scope_narrow → renumber
#
# Invariant: each v1 clause receives at MOST one transformation from the
# 4 content passes. Structural transformations (depr/split) consume the
# clause before content passes ever see it.


# Phrase pairs for strengthening. Picked to be unambiguous and easy to detect
# in v1 prose. The LLM generates v1 using these phrases naturally because the
# system prompt explicitly says "Use 'shall' for mandatory, 'should' for
# recommendations, 'may' for permissions".
STRENGTHEN_PATTERNS: list[tuple[str, str]] = [
    ("should", "shall"),
    ("may", "shall"),
    ("is recommended", "is required"),
    ("are recommended", "are required"),
    ("where practical", "at all times"),
    ("where feasible", "at all times"),
]

# Scope-narrowing modifier additions. Applied as text insertion BEFORE a
# target generic noun phrase in the clause body.
SCOPE_NARROW_PATTERNS: list[tuple[str, str]] = [
    ("all equipment", "all Class-B rotating equipment"),
    ("any equipment", "any Class-B rotating equipment"),
    ("equipment", "Class-B rotating equipment"),
    ("all operators", "all qualified operators"),
    ("all personnel", "all qualified personnel"),
    ("all records", "all controlled records"),
]

# Standard-level terminology updates for v1→v2.
# Generic equipment names that change to reflect technology evolution.
# These replace the old equipment-ID-based term_replace (IDs now live
# only in WIs, not in the standard text).
# Both cased forms are listed so sentence-initial occurrences are caught.
STANDARD_TERM_REPLACEMENTS: list[tuple[str, str]] = [
    ("Hydraulic press", "Servo-hydraulic press"),
    ("hydraulic press", "servo-hydraulic press"),
    ("Horizontal CNC machining center", "5-axis CNC machining center"),
    ("horizontal CNC machining center", "5-axis CNC machining center"),
]


def _clone_clause(cl: Clause, new_id: str | None = None, new_body: str | None = None, new_heading: str | None = None) -> Clause:
    """Return a new Clause with optional field overrides."""
    return Clause(
        clause_id=new_id if new_id is not None else cl.clause_id,
        heading=new_heading if new_heading is not None else cl.heading,
        body=new_body if new_body is not None else cl.body,
    )


def _eligible_for_strengthen(cl: Clause) -> str | None:
    """Return the v1 phrase to replace, or None if no pattern matches."""
    for from_phrase, _ in STRENGTHEN_PATTERNS:
        if from_phrase in cl.body:
            return from_phrase
    return None


def _eligible_for_scope_narrow(cl: Clause) -> str | None:
    for from_phrase, _ in SCOPE_NARROW_PATTERNS:
        if from_phrase in cl.body:
            return from_phrase
    return None


def _deep_copy_standard(std: Standard, new_version: Literal["v1", "v2"]) -> Standard:
    """Deep copy a Standard with a version tag override."""
    new_chapters: list[Chapter] = []
    for ch in std.chapters:
        new_sections: list[Section] = []
        for sec in ch.sections:
            new_clauses = [_clone_clause(cl) for cl in sec.clauses]
            new_sections.append(
                Section(section_number=sec.section_number, title=sec.title, clauses=new_clauses)
            )
        new_chapters.append(
            Chapter(chapter_number=ch.chapter_number, title=ch.title, sections=new_sections)
        )
    return Standard(version=new_version, title=std.title, chapters=new_chapters)


def _section_of(std: Standard, clause_id: str) -> Section | None:
    """Find the section containing a given clause_id."""
    for ch in std.chapters:
        for sec in ch.sections:
            if any(cl.clause_id == clause_id for cl in sec.clauses):
                return sec
    return None


def _remove_clause(std: Standard, clause_id: str) -> Clause | None:
    """Remove a clause from the standard in-place. Returns the removed clause."""
    for ch in std.chapters:
        for sec in ch.sections:
            for i, cl in enumerate(sec.clauses):
                if cl.clause_id == clause_id:
                    return sec.clauses.pop(i)
    return None


def _renumber_section_locally(sec: Section, chapter_number: int) -> dict[str, str]:
    """Re-assign sequential clause numbers within a section.

    Returns a dict mapping old clause_id → new clause_id for any clauses
    whose ID changed. Used to rebuild references after structural edits.
    """
    remap: dict[str, str] = {}
    for i, cl in enumerate(sec.clauses, start=1):
        expected_id = f"{chapter_number}.{sec.section_number}.{i}"
        if cl.clause_id != expected_id:
            remap[cl.clause_id] = expected_id
            cl.clause_id = expected_id
    return remap


# --- Pass 1: deprecate --------------------------------------------------------


def _apply_deprecations(
    v2: Standard,
    affected: set[str],
    rng: random.Random,
    log: TransformationLog,
    n: int = 4,
) -> set[str]:
    """Remove 4 clauses from v2. Returns the set of newly-affected clause IDs.

    Picks from clauses not yet affected by any transformation. After removal,
    local renumbering of each touched section brings the surviving clause IDs
    back to sequential form.
    """
    all_ids = [cl.clause_id for cl in v2.all_clauses() if cl.clause_id not in affected]
    targets = rng.sample(all_ids, k=n)

    touched_sections: list[tuple[int, Section]] = []
    for clause_id in targets:
        sec = _section_of(v2, clause_id)
        if sec is None:
            continue
        ch = next(c for c in v2.chapters if sec in c.sections)
        removed = _remove_clause(v2, clause_id)
        assert removed is not None
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.DEPRECATE,
                v1_clause_ids=[clause_id],
                v2_clause_ids=[],
                payload={"removed_heading": removed.heading},
            )
        )
        touched_sections.append((ch.chapter_number, sec))

    # Local renumber each touched section so surviving clauses are sequential.
    for chap_num, sec in touched_sections:
        remap = _renumber_section_locally(sec, chap_num)
        for old, new in remap.items():
            log.entries.append(
                TransformationEntry(
                    transformation_type=TransformationType.RENUMBER,
                    v1_clause_ids=[old],
                    v2_clause_ids=[new],
                    payload={"reason": "local_shift_after_deprecate"},
                )
            )

    return set(targets)


# --- Pass 2: clause split ----------------------------------------------------


def _apply_clause_splits(
    v2: Standard,
    affected: set[str],
    rng: random.Random,
    log: TransformationLog,
    n: int = 3,
) -> set[str]:
    """Split 3 v1 clauses into 2 v2 clauses each.

    Split mechanics: takes the v1 clause body, splits it on the first
    sentence boundary after the midpoint, assigns each half to a new
    clause. The original clause's ID is retained for part A. Part B gets
    the next sequential ID; all later siblings in the section shift down.
    """
    candidates = [cl.clause_id for cl in v2.all_clauses() if cl.clause_id not in affected]
    targets = rng.sample(candidates, k=n)
    newly_affected: set[str] = set()

    for clause_id in targets:
        sec = _section_of(v2, clause_id)
        if sec is None:
            continue
        ch = next(c for c in v2.chapters if sec in c.sections)

        idx = next(i for i, cl in enumerate(sec.clauses) if cl.clause_id == clause_id)
        original = sec.clauses[idx]

        # Split body on sentence boundary
        sentences = _split_sentences(original.body)
        if len(sentences) < 2:
            words = original.body.split()
            mid = len(words) // 2
            first_half = " ".join(words[:mid]) + "."
            second_half = " ".join(words[mid:])
        else:
            mid = len(sentences) // 2
            first_half = " ".join(sentences[:mid])
            second_half = " ".join(sentences[mid:])

        # Shift all later siblings down by 1 position in their clause_id
        # BEFORE we insert part B, so part B can take the cleared slot.
        # Iterate in reverse to avoid collisions.
        shift_remap: dict[str, str] = {}
        for i in range(len(sec.clauses) - 1, idx, -1):
            sib = sec.clauses[i]
            old = sib.clause_id
            new_n = int(old.split(".")[-1]) + 1
            new = f"{ch.chapter_number}.{sec.section_number}.{new_n}"
            sib.clause_id = new
            shift_remap[old] = new

        # Part A keeps the original ID
        part_a = Clause(
            clause_id=original.clause_id,
            heading=original.heading,
            body=first_half,
        )
        # Part B gets the original ID's position + 1
        part_b_n = int(original.clause_id.split(".")[-1]) + 1
        part_b_id = f"{ch.chapter_number}.{sec.section_number}.{part_b_n}"
        part_b = Clause(
            clause_id=part_b_id,
            heading=original.heading + " (continued)",
            body=second_half,
        )

        sec.clauses[idx] = part_a
        sec.clauses.insert(idx + 1, part_b)

        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.CLAUSE_SPLIT,
                v1_clause_ids=[clause_id],
                v2_clause_ids=[part_a.clause_id, part_b.clause_id],
                payload={"note": "body halved on sentence boundary"},
            )
        )
        # Log sibling shifts as renumbers
        for old, new in shift_remap.items():
            log.entries.append(
                TransformationEntry(
                    transformation_type=TransformationType.RENUMBER,
                    v1_clause_ids=[old],
                    v2_clause_ids=[new],
                    payload={"reason": "local_shift_after_split"},
                )
            )

        newly_affected.add(clause_id)

    return newly_affected


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — splits on '. ', '? ', '! '.

    Adequate for standard-document prose which avoids abbreviations. The
    trailing period is re-attached to each sentence.
    """
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# --- Pass 3: clause insertions ----------------------------------------------


def _apply_clause_insertions(
    v2: Standard,
    rng: random.Random,
    log: TransformationLog,
    n: int = 5,
) -> None:
    """Insert 5 brand-new clauses into v2 (no v1 counterpart).

    Each insertion picks a random section and appends a new clause at the
    end with the next sequential ID. Bodies are drawn from a curated pool
    of generic v2-era additions (phrased to sound like standard updates).
    """
    insertion_templates = [
        (
            "Cybersecurity Controls",
            "Equipment connected to the site network shall be registered in the IT asset inventory "
            "prior to operational release. Access credentials shall be issued per the site identity "
            "management procedure and reviewed quarterly. Firmware updates shall be applied within "
            "30 days of vendor release unless an approved deferral is recorded.",
        ),
        (
            "Environmental Impact Assessment",
            "New equipment introductions shall include an environmental impact statement covering "
            "energy consumption, coolant and lubricant waste streams, and noise exposure. The "
            "statement shall be reviewed by the facility sustainability lead prior to installation "
            "qualification.",
        ),
        (
            "Supplier Change Notification",
            "Changes to approved suppliers for critical consumables shall be notified in writing at "
            "least 45 days before effect. A containment action may be imposed if the receiving "
            "process is unable to validate the new source within the notification window.",
        ),
        (
            "Data Integrity Requirements",
            "Records generated by automated inspection equipment shall be stored in tamper-evident "
            "form with an audit trail showing user, timestamp, and any value modifications. "
            "Manual overrides of inspection results shall be traceable to a qualified technician "
            "and accompanied by a documented justification.",
        ),
        (
            "Digital Work Instruction Rollout",
            "Work Instructions distributed through the digital authoring system shall include a "
            "hash-based version identifier visible on every printed page. Offline copies older than "
            "90 days shall not be used for production operations without verification against the "
            "current digital version.",
        ),
    ]
    picks = rng.sample(insertion_templates, k=n)

    # Choose n distinct sections to receive insertions
    all_sections: list[tuple[int, Section]] = []
    for ch in v2.chapters:
        for sec in ch.sections:
            all_sections.append((ch.chapter_number, sec))
    chosen_sections = rng.sample(all_sections, k=n)

    for (chap_num, sec), (heading, body) in zip(chosen_sections, picks):
        next_n = len(sec.clauses) + 1
        new_id = f"{chap_num}.{sec.section_number}.{next_n}"
        new_cl = Clause(clause_id=new_id, heading=heading, body=body)
        sec.clauses.append(new_cl)
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.CLAUSE_INSERT,
                v1_clause_ids=[],
                v2_clause_ids=[new_id],
                payload={"heading": heading},
            )
        )


# --- Pass 4a: term replacement ----------------------------------------------


def _apply_term_replacements(
    v2: Standard,
    affected: set[str],
    log: TransformationLog,
) -> set[str]:
    """Replace standard-level terminology that changed between v1 and v2.

    Operates on generic equipment names (e.g. "hydraulic press" →
    "servo-hydraulic press"), NOT on equipment asset IDs. Asset IDs
    live only in Work Instructions and are handled by the glossary.

    Deterministic: every clause containing an old term gets the term
    replaced. Number of affected clauses ≈ 9 (4 hydraulic + 5 CNC).
    """
    newly_affected: set[str] = set()

    for cl in v2.all_clauses():
        if cl.clause_id in affected:
            continue
        changed = False
        for old_term, new_term in STANDARD_TERM_REPLACEMENTS:
            if old_term in cl.body:
                cl.body = cl.body.replace(old_term, new_term)
                log.entries.append(
                    TransformationEntry(
                        transformation_type=TransformationType.TERM_REPLACE,
                        v1_clause_ids=[cl.clause_id],
                        v2_clause_ids=[cl.clause_id],
                        payload={"from": old_term, "to": new_term},
                    )
                )
                changed = True
        if changed:
            newly_affected.add(cl.clause_id)

    return newly_affected


# --- Pass 4b: strengthening -------------------------------------------------


def _apply_strengthenings(
    v2: Standard,
    affected: set[str],
    rng: random.Random,
    log: TransformationLog,
    n: int = 8,
) -> set[str]:
    """Tighten modal verbs in 8 eligible clauses."""
    eligible = [
        cl for cl in v2.all_clauses()
        if cl.clause_id not in affected and _eligible_for_strengthen(cl) is not None
    ]
    if len(eligible) < n:
        print(f"  [warn] strengthening: only {len(eligible)} eligible clauses, target was {n}")
        n = len(eligible)
    targets = rng.sample(eligible, k=n)
    newly_affected: set[str] = set()

    for cl in targets:
        from_phrase = _eligible_for_strengthen(cl)
        if from_phrase is None:
            continue
        to_phrase = next(tp for fp, tp in STRENGTHEN_PATTERNS if fp == from_phrase)
        cl.body = cl.body.replace(from_phrase, to_phrase, 1)
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.STRENGTHEN,
                v1_clause_ids=[cl.clause_id],
                v2_clause_ids=[cl.clause_id],
                payload={"from_phrase": from_phrase, "to_phrase": to_phrase},
            )
        )
        newly_affected.add(cl.clause_id)

    return newly_affected


# --- Pass 4c: scope narrowing -----------------------------------------------


def _apply_scope_narrowings(
    v2: Standard,
    affected: set[str],
    rng: random.Random,
    log: TransformationLog,
    n: int = 5,
) -> set[str]:
    """Narrow scope in 5 eligible clauses by replacing generic noun phrases."""
    eligible = [
        cl for cl in v2.all_clauses()
        if cl.clause_id not in affected and _eligible_for_scope_narrow(cl) is not None
    ]
    if len(eligible) < n:
        print(f"  [warn] scope narrow: only {len(eligible)} eligible clauses, target was {n}")
        n = len(eligible)
    targets = rng.sample(eligible, k=n)
    newly_affected: set[str] = set()

    for cl in targets:
        from_phrase = _eligible_for_scope_narrow(cl)
        if from_phrase is None:
            continue
        to_phrase = next(tp for fp, tp in SCOPE_NARROW_PATTERNS if fp == from_phrase)
        cl.body = cl.body.replace(from_phrase, to_phrase, 1)
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.SCOPE_NARROW,
                v1_clause_ids=[cl.clause_id],
                v2_clause_ids=[cl.clause_id],
                payload={"from_scope": from_phrase, "to_scope": to_phrase},
            )
        )
        newly_affected.add(cl.clause_id)

    return newly_affected


# --- Pass 4d: targeted renumbering ------------------------------------------


def _apply_targeted_renumbers(
    v2: Standard,
    affected: set[str],
    rng: random.Random,
    log: TransformationLog,
    n: int = 8,
) -> set[str]:
    """Move 8 clauses across sections — same content, new ID.

    This is the 'renumbering-only' transformation type from the budget.
    It's on top of the incidental local renumbering that structural passes
    trigger. Each targeted renumber moves a clause to a different section
    within the same chapter (cross-chapter moves would be semantically
    implausible).
    """
    eligible = [cl for cl in v2.all_clauses() if cl.clause_id not in affected]
    if len(eligible) < n:
        print(f"  [warn] renumber: only {len(eligible)} eligible clauses, target was {n}")
        n = len(eligible)
    targets = rng.sample(eligible, k=n)
    newly_affected: set[str] = set()

    for cl in targets:
        old_id = cl.clause_id
        chap_num = int(old_id.split(".")[0])
        current_sec_num = int(old_id.split(".")[1])
        # Pick a different section in the same chapter
        chapter = next(ch for ch in v2.chapters if ch.chapter_number == chap_num)
        other_sections = [s for s in chapter.sections if s.section_number != current_sec_num]
        target_sec = rng.choice(other_sections)

        # Remove from current section
        current_sec = _section_of(v2, old_id)
        assert current_sec is not None
        current_sec.clauses = [c for c in current_sec.clauses if c.clause_id != old_id]

        # Append to target section with new ID
        new_n = len(target_sec.clauses) + 1
        new_id = f"{chap_num}.{target_sec.section_number}.{new_n}"
        cl.clause_id = new_id
        target_sec.clauses.append(cl)

        # Locally renumber the source section (sibling IDs shift down)
        remap = _renumber_section_locally(current_sec, chap_num)

        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.RENUMBER,
                v1_clause_ids=[old_id],
                v2_clause_ids=[new_id],
                payload={"reason": "targeted_move"},
            )
        )
        for old, new in remap.items():
            log.entries.append(
                TransformationEntry(
                    transformation_type=TransformationType.RENUMBER,
                    v1_clause_ids=[old],
                    v2_clause_ids=[new],
                    payload={"reason": "local_shift_after_targeted_renumber"},
                )
            )

        newly_affected.add(old_id)

    return newly_affected


# --- Orchestrator -----------------------------------------------------------


def transform_to_v2(
    v1: Standard,
    glossary: list[dict],
    rng: random.Random,
) -> tuple[Standard, TransformationLog]:
    """Apply all mechanical transformations from the budget to produce v2.

    Returns (v2_standard, transformation_log). Pure function modulo the
    RNG seed and the glossary — no LLM calls, no filesystem I/O.

    Mutation policy: v1 is not modified. v2 is a deep copy that is mutated
    in place by each pass. The log is the authoritative record of every
    change; rendering and ground-truth computation consume it.
    """
    v2 = _deep_copy_standard(v1, new_version="v2")
    log = TransformationLog(entries=[])
    affected: set[str] = set()

    # Pass 1: deprecate (structural)
    affected |= _apply_deprecations(v2, affected, rng, log, n=4)

    # Pass 2: clause splits (structural)
    affected |= _apply_clause_splits(v2, affected, rng, log, n=3)

    # Pass 3: clause insertions (structural, no v1 clauses consumed)
    _apply_clause_insertions(v2, rng, log, n=5)

    # Pass 4a: term replacement (applied to all clauses with matching terms,
    # independent of the 'affected' set — the budget number is implicit)
    affected |= _apply_term_replacements(v2, affected, log)

    # Pass 4b/c/d: content changes on remaining unchanged pool
    affected |= _apply_strengthenings(v2, affected, rng, log, n=8)
    affected |= _apply_scope_narrowings(v2, affected, rng, log, n=5)
    affected |= _apply_targeted_renumbers(v2, affected, rng, log, n=8)

    return v2, log


# ---------------------------------------------------------------------------
# apply_semantic_changes() — 12 hand-labeled semantic mutations
# ---------------------------------------------------------------------------

# Pinned targets — identified by inspecting v1 output.
SEMANTIC_TARGETS: dict = {
    "tone_shift":      ["1.1.1", "2.1.2", "3.1.1"],
    "xref_chain":      ["1.1.3", "1.1.4", "1.2.1"],
    "ambiguous_scope": ["1.2.4", "1.2.3", "4.2.1"],
    "clause_merge":    [("1.5.1", "1.5.2"), ("2.2.1", "2.2.2"), ("4.4.1", "4.4.2")],
}

# Hedging → imperative substitutions for tone_shift mutations.
_TONE_SHIFT_REPLACEMENTS: list[tuple[str, str]] = [
    ("are expected to be documented and reviewed", "shall be documented and reviewed"),
    ("is expected to be maintained", "shall be maintained"),
    ("is expected to be conducted", "shall be conducted"),
    ("are expected to verify", "shall verify"),
    ("are expected to confirm", "shall confirm"),
    ("are expected to ensure", "shall ensure"),
    ("are expected to maintain", "shall maintain"),
    ("are expected to conduct", "shall conduct"),
    ("are expected to", "shall"),
    ("is expected to", "shall"),
    ("it is advisable to", ""),
    ("it is advisable that", ""),
    ("it is recommended that", ""),
    ("it is recommended to", ""),
    ("should consider", "shall"),
    ("is encouraged to", "shall"),
    ("where appropriate, ", ""),
    ("where applicable, ", ""),
]

# Scope qualifiers to blur for ambiguous_scope mutations.
_SCOPE_BLUR_PATTERNS: list[tuple[str, str]] = [
    ("every 30 days", "periodically"),
    ("every 90 days", "periodically"),
    ("every six months", "periodically"),
    ("at least once per production shift", "at regular intervals"),
    ("monthly", "periodically"),
    ("quarterly", "periodically"),
    ("annually", "periodically"),
    ("all Class-B rotating equipment", "all applicable equipment"),
    ("Class-B rotating equipment", "applicable equipment"),
    ("all controlled records", "all records"),
    ("all qualified operators", "all operators"),
    ("all qualified personnel", "all personnel"),
    ("all personnel", "relevant personnel"),
]


def apply_semantic_changes(
    v2: Standard,
    log: TransformationLog,
    semantic_targets: dict | None = None,
) -> None:
    """Apply 12 semantic mutations to pinned target clauses in v2.

    Mutates v2 in-place and appends semantic=True entries to the log.
    compute_expected_edits() ignores semantic entries — they feed the
    scaffold_semantic_changes() hand-labeling step only.
    """
    if semantic_targets is None:
        semantic_targets = SEMANTIC_TARGETS

    # --- Tone shifts: remove hedging language ---
    for cid in semantic_targets.get("tone_shift", []):
        cl = v2.clause_by_id(cid)
        if cl is None:
            print(f"  [warn] semantic tone_shift: clause {cid} not found in v2")
            continue
        applied = False
        for from_phrase, to_phrase in _TONE_SHIFT_REPLACEMENTS:
            if from_phrase in cl.body:
                cl.body = cl.body.replace(from_phrase, to_phrase, 1).strip()
                applied = True
                break
        if not applied:
            print(f"  [warn] tone_shift {cid}: no hedging pattern matched")
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.SEMANTIC_TONE_SHIFT,
                v1_clause_ids=[cid],
                v2_clause_ids=[cid],
                payload={
                    "description": (
                        "Hedging language removed; enforceability tightened "
                        "without an explicit stronger modal verb"
                    )
                },
            )
        )

    # --- Cross-reference chain breakages: no text change ---
    for cid in semantic_targets.get("xref_chain", []):
        if v2.clause_by_id(cid) is None:
            print(f"  [warn] semantic xref_chain: clause {cid} not found in v2")
            continue
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.SEMANTIC_XREF_CHAIN,
                v1_clause_ids=[cid],
                v2_clause_ids=[cid],
                payload={
                    "description": (
                        "Clause text unchanged but references a clause that was "
                        "deprecated or structurally moved; effective meaning "
                        "altered via chain breakage"
                    )
                },
            )
        )

    # --- Ambiguous scope shifts ---
    for cid in semantic_targets.get("ambiguous_scope", []):
        cl = v2.clause_by_id(cid)
        if cl is None:
            print(f"  [warn] semantic ambiguous_scope: clause {cid} not found in v2")
            continue
        applied = False
        for from_phrase, to_phrase in _SCOPE_BLUR_PATTERNS:
            if from_phrase in cl.body:
                cl.body = cl.body.replace(from_phrase, to_phrase, 1)
                applied = True
                break
        if not applied:
            print(f"  [warn] ambiguous_scope {cid}: no scope pattern matched")
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.SEMANTIC_AMBIGUOUS_SCOPE,
                v1_clause_ids=[cid],
                v2_clause_ids=[cid],
                payload={
                    "description": (
                        "Precise scope qualifier replaced with vague term; "
                        "interpretation space widened"
                    )
                },
            )
        )

    # --- Clause merges: A absorbs B ---
    for pair in semantic_targets.get("clause_merge", []):
        id_a, id_b = pair
        cl_a = v2.clause_by_id(id_a)
        cl_b = v2.clause_by_id(id_b)
        if cl_a is None or cl_b is None:
            print(f"  [warn] semantic clause_merge: {id_a} or {id_b} not in v2")
            continue
        # Merge: A absorbs B with a transition sentence
        cl_a.body = (
            cl_a.body.rstrip(". ")
            + ". Additionally, "
            + cl_b.body[0].lower()
            + cl_b.body[1:]
        )
        cl_a.heading = cl_a.heading + " and " + cl_b.heading.lower()
        # Remove B from its section
        for ch in v2.chapters:
            for sec in ch.sections:
                sec.clauses = [c for c in sec.clauses if c.clause_id != id_b]
        log.entries.append(
            TransformationEntry(
                transformation_type=TransformationType.SEMANTIC_CLAUSE_MERGE,
                v1_clause_ids=[id_a, id_b],
                v2_clause_ids=[id_a],
                payload={
                    "description": (
                        f"Clauses {id_a} and {id_b} merged into {id_a}; "
                        f"{id_b} removed from v2"
                    )
                },
            )
        )


# ---------------------------------------------------------------------------
# build_wis() — generate 30 Work Instructions via LLM
# ---------------------------------------------------------------------------

# Curated procedure topics, ~7 per chapter domain.
WI_TOPICS: list[dict[str, str]] = [
    # Chapter 1 — General Provisions
    {"topic": "Document Control for Quality Records", "domain": "general"},
    {"topic": "Internal Audit Preparation Procedure", "domain": "general"},
    {"topic": "Employee Onboarding for Quality Systems", "domain": "general"},
    {"topic": "Management Review Data Collection", "domain": "general"},
    {"topic": "Corrective Action Request Processing", "domain": "general"},
    {"topic": "Supplier Qualification Documentation", "domain": "general"},
    # Chapter 2 — Equipment Qualification
    {"topic": "CNC Machining Center Setup and Alignment", "domain": "equipment"},
    {"topic": "Surface Grinder Wheel Dressing Procedure", "domain": "equipment"},
    {"topic": "Vertical Turning Lathe Spindle Calibration", "domain": "equipment"},
    {"topic": "Hydraulic Press Die Changeover", "domain": "equipment"},
    {"topic": "Injection Molding Unit Purging Procedure", "domain": "equipment"},
    {"topic": "Equipment Decommissioning Checklist", "domain": "equipment"},
    {"topic": "Coordinate Measuring Machine Probe Qualification", "domain": "equipment"},
    # Chapter 3 — Operational Controls
    {"topic": "Pre-shift Equipment Inspection Checklist", "domain": "operations"},
    {"topic": "Torque Wrench Calibration Verification", "domain": "operations"},
    {"topic": "Fixture Setup for Multi-part Runs", "domain": "operations"},
    {"topic": "Coolant Concentration Monitoring", "domain": "operations"},
    {"topic": "Lockout-Tagout for Press Maintenance", "domain": "operations"},
    {"topic": "Tool Wear Monitoring and Replacement", "domain": "operations"},
    {"topic": "Process Parameter Adjustment Procedure", "domain": "operations"},
    # Chapter 4 — Inspection & Metrology
    {"topic": "First Article Inspection Execution", "domain": "inspection"},
    {"topic": "In-Process Dimensional Check Procedure", "domain": "inspection"},
    {"topic": "Final Inspection and Release Authorization", "domain": "inspection"},
    {"topic": "Gauge R&R Study Execution", "domain": "inspection"},
    {"topic": "CMM Program Validation Procedure", "domain": "inspection"},
    {"topic": "Visual Inspection Criteria for Surface Defects", "domain": "inspection"},
    # Chapter 5 — Nonconformance & CAPA
    {"topic": "Nonconforming Material Quarantine Procedure", "domain": "nonconformance"},
    {"topic": "Root Cause Analysis Using 5-Why Method", "domain": "nonconformance"},
    {"topic": "Corrective Action Implementation and Tracking", "domain": "nonconformance"},
    {"topic": "NCR Closure and Effectiveness Verification", "domain": "nonconformance"},
]

# Map each domain to the chapter numbers it naturally references.
_DOMAIN_TO_CHAPTERS: dict[str, list[int]] = {
    "general":        [1, 2],
    "equipment":      [2, 3],
    "operations":     [3, 2],
    "inspection":     [4, 3],
    "nonconformance": [5, 4],
}


def _pick_clause_refs(
    v1: Standard,
    log: TransformationLog,
    domain: str,
    length_bucket: LengthBucket,
    rng: random.Random,
) -> list[dict]:
    """Select seeded clause references for one WI.

    Returns list of dicts: {"clause_id", "is_edit_requiring"}.
    For long WIs, ensures edit-requiring refs span top/middle/bottom.
    """
    # Separate clauses into edit-requiring vs unchanged
    affected_ids = set()
    for e in log.mechanical_entries():
        affected_ids.update(e.v1_clause_ids)

    primary_ch, secondary_ch = _DOMAIN_TO_CHAPTERS[domain]
    all_v1 = v1.all_clauses()

    domain_clauses = [
        c for c in all_v1
        if int(c.clause_id.split(".")[0]) in (primary_ch, secondary_ch)
    ]
    edit_pool = [c for c in domain_clauses if c.clause_id in affected_ids]
    safe_pool = [c for c in domain_clauses if c.clause_id not in affected_ids]

    # Fallback: if pools are thin, widen to all chapters
    if len(edit_pool) < 2:
        edit_pool = [c for c in all_v1 if c.clause_id in affected_ids]
    if len(safe_pool) < 3:
        safe_pool = [c for c in all_v1 if c.clause_id not in affected_ids]

    # Target ref counts by bucket
    if length_bucket == LengthBucket.SHORT:
        n_edit, n_safe = 1, rng.randint(2, 3)
    elif length_bucket == LengthBucket.MEDIUM:
        n_edit, n_safe = 2, rng.randint(2, 4)
    else:  # LONG
        n_edit, n_safe = 3, rng.randint(3, 5)

    picked_edit = rng.sample(edit_pool, min(n_edit, len(edit_pool)))
    picked_safe = rng.sample(safe_pool, min(n_safe, len(safe_pool)))

    refs = []
    for c in picked_edit:
        refs.append({"clause_id": c.clause_id, "is_edit_requiring": True})
    for c in picked_safe:
        refs.append({"clause_id": c.clause_id, "is_edit_requiring": False})

    rng.shuffle(refs)
    return refs


def _section_skeleton(length_bucket: LengthBucket, rng: random.Random) -> list[dict]:
    """Return a section skeleton with section headings and target word counts."""
    base = [
        {"heading": "Purpose", "target_words": 60},
        {"heading": "Scope", "target_words": 50},
    ]
    if length_bucket == LengthBucket.SHORT:
        body = [{"heading": "Procedure", "target_words": 200}]
    elif length_bucket == LengthBucket.MEDIUM:
        body = [
            {"heading": "Prerequisites", "target_words": 80},
            {"heading": "Procedure", "target_words": 250},
            {"heading": "Verification", "target_words": 80},
        ]
    else:  # LONG
        n_sub = rng.randint(3, 5)
        body = [
            {"heading": "Prerequisites", "target_words": 100},
        ]
        for i in range(1, n_sub + 1):
            body.append(
                {"heading": f"Procedure Step {i}", "target_words": 150}
            )
        body.append({"heading": "Verification", "target_words": 100})
        body.append({"heading": "Records", "target_words": 80})

    return base + body


def _compute_position_bucket(
    section_index: int, total_sections: int
) -> PositionBucket:
    """Normalize a section index to [0,1] and bucket."""
    if total_sections <= 1:
        return PositionBucket.MIDDLE
    pos = section_index / (total_sections - 1)
    if pos < 0.33:
        return PositionBucket.TOP
    elif pos < 0.67:
        return PositionBucket.MIDDLE
    else:
        return PositionBucket.BOTTOM


def build_wis(
    v1: Standard,
    log: TransformationLog,
    glossary: list[dict],
    rng: random.Random,
    n: int = 30,
    verbose: bool = False,
) -> list[WorkInstruction]:
    """Generate n Work Instructions via LLM.

    One LLM call per WI (gpt-4o-mini, temperature=0.3).
    Returns a list of WorkInstruction Pydantic objects.
    """
    client = get_client()

    # Assign length buckets: 10 short, 12 medium, 8 long
    buckets = (
        [LengthBucket.SHORT] * 10
        + [LengthBucket.MEDIUM] * 12
        + [LengthBucket.LONG] * 8
    )
    rng.shuffle(buckets)

    # Shuffle topics
    topics = WI_TOPICS[:n]
    rng.shuffle(topics)

    # Build equipment-ID lookup for WI prose (these DO appear in WIs)
    eq_terms = [e for e in glossary if e["category"] == "equipment_id"]
    abbrev_terms = [e for e in glossary if e["category"] == "abbreviation"]

    wis: list[WorkInstruction] = []

    for i in range(n):
        wi_id = f"WI-{i + 1:03d}"
        topic_info = topics[i]
        bucket = buckets[i]
        refs = _pick_clause_refs(v1, log, topic_info["domain"], bucket, rng)
        skeleton = _section_skeleton(bucket, rng)

        # Build the prompt
        ref_lines = "\n".join(
            f"  - AIQS {r['clause_id']}" for r in refs
        )
        skeleton_lines = "\n".join(
            f"  ## {j+1}. {s['heading']}  (~{s['target_words']} words)"
            for j, s in enumerate(skeleton)
        )
        # Pick 2-3 equipment terms to weave into the WI
        wi_eq = rng.sample(eq_terms, min(2, len(eq_terms)))
        eq_line = ", ".join(
            f"{e['term']} ({e['description']})" for e in wi_eq
        )
        abbrev_line = ", ".join(
            f"{e['term']} = {e['description']}" for e in abbrev_terms
        )

        system_prompt = (
            "You are a technical writer for an industrial manufacturing plant. "
            "Write realistic, formal Work Instructions that reference the Acme "
            "Industrial Quality Standard (AIQS).\n\n"
            "Rules:\n"
            "- Use ONLY the AIQS clause IDs provided. Never invent clause IDs.\n"
            "- Cite clauses in the format 'AIQS X.Y.Z' (e.g. 'per AIQS 3.2.1').\n"
            "- Each provided clause ID must appear at least once in the text.\n"
            "- Use equipment asset IDs exactly as given (e.g. 2847.310.0042).\n"
            "- Use abbreviations naturally, expanding on first use.\n"
            "- Write in imperative/shall style typical of manufacturing SOPs.\n"
            "- Return ONLY the Markdown text of the WI. No preamble."
        )

        user_prompt = (
            f"Write Work Instruction {wi_id}: {topic_info['topic']}\n\n"
            f"Length: {bucket.value} ({skeleton_lines.count('##')} sections)\n\n"
            f"AIQS clause references to include:\n{ref_lines}\n\n"
            f"Section structure:\n{skeleton_lines}\n\n"
            f"Equipment to reference: {eq_line}\n"
            f"Abbreviations to use: {abbrev_line}\n\n"
            f"Begin the document with:\n"
            f"# {wi_id}: {topic_info['topic']}\n"
        )

        if verbose:
            print(f"  [{i+1}/{n}] {wi_id} ({bucket.value}) — {topic_info['topic']}")

        # LLM call with retry
        body_markdown = ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                body_markdown = resp.choices[0].message.content.strip()

                # Validate: every seeded clause ID must appear
                missing = [
                    r["clause_id"]
                    for r in refs
                    if f"AIQS {r['clause_id']}" not in body_markdown
                ]
                if missing:
                    raise ValueError(
                        f"Missing clause refs: {missing}"
                    )
                break
            except Exception as exc:
                if verbose:
                    print(f"    attempt {attempt+1} failed: {exc}")
                if attempt == 2:
                    print(f"    ⚠ {wi_id}: giving up after 3 attempts")

        # Build structured references with position buckets
        total_sections = len(skeleton)
        structured_refs: list[ClauseReference] = []
        for r in refs:
            # Find which section the ref appears in
            aiqs_tag = f"AIQS {r['clause_id']}"
            section_idx = 0
            # Rough heuristic: split body by "## " headers, find which chunk
            chunks = body_markdown.split("## ")
            for ci, chunk in enumerate(chunks):
                if aiqs_tag in chunk:
                    section_idx = ci
                    break
            pos = _compute_position_bucket(section_idx, total_sections)
            structured_refs.append(
                ClauseReference(
                    clause_id=r["clause_id"],
                    position_bucket=pos,
                    is_edit_requiring=r["is_edit_requiring"],
                )
            )

        wi = WorkInstruction(
            wi_id=wi_id,
            title=topic_info["topic"],
            length_bucket=bucket,
            topic=topic_info["topic"],
            references=structured_refs,
            body_markdown=body_markdown,
        )
        wis.append(wi)

    if verbose:
        print(f"\n  Generated {len(wis)} Work Instructions")
        edit_refs = sum(
            1 for w in wis for r in w.references if r.is_edit_requiring
        )
        total_refs = sum(len(w.references) for w in wis)
        print(f"  Total refs: {total_refs}, edit-requiring: {edit_refs}")

    return wis


# ---------------------------------------------------------------------------
# compute_expected_edits() — deterministic ground truth from mechanical log
# ---------------------------------------------------------------------------


def compute_expected_edits(
    wis: list[WorkInstruction],
    log: TransformationLog,
) -> list[dict]:
    """Compute expected edit behavior for every clause reference in every WI.

    Only considers mechanical log entries. Semantic entries are excluded
    (handled via scaffold_semantic_changes for hand-labeling).

    Returns list of dicts: {
        "wi_id", "clause_id", "position_bucket",
        "expected_behavior": "edit_required" | "flag_for_review" | "no_action_required",
        "transformation_type": str | null,
        "detail": str
    }
    """
    # Build lookup: v1_clause_id → list of mechanical entries
    v1_to_entries: dict[str, list[TransformationEntry]] = {}
    for entry in log.mechanical_entries():
        for cid in entry.v1_clause_ids:
            v1_to_entries.setdefault(cid, []).append(entry)

    results: list[dict] = []

    for wi in wis:
        for ref in wi.references:
            entries = v1_to_entries.get(ref.clause_id, [])

            if not entries:
                results.append({
                    "wi_id": wi.wi_id,
                    "clause_id": ref.clause_id,
                    "position_bucket": ref.position_bucket.value,
                    "expected_behavior": "no_action_required",
                    "transformation_type": None,
                    "detail": "Clause unchanged between v1 and v2",
                })
                continue

            # Use the first (most impactful) transformation
            entry = entries[0]
            t = entry.transformation_type

            if t == TransformationType.DEPRECATE:
                results.append({
                    "wi_id": wi.wi_id,
                    "clause_id": ref.clause_id,
                    "position_bucket": ref.position_bucket.value,
                    "expected_behavior": "flag_for_review",
                    "transformation_type": t.value,
                    "detail": (
                        f"Clause {ref.clause_id} deprecated in v2; "
                        "WI reference must be reviewed for removal or replacement"
                    ),
                })
            elif t == TransformationType.RENUMBER:
                new_id = entry.v2_clause_ids[0] if entry.v2_clause_ids else "?"
                results.append({
                    "wi_id": wi.wi_id,
                    "clause_id": ref.clause_id,
                    "position_bucket": ref.position_bucket.value,
                    "expected_behavior": "edit_required",
                    "transformation_type": t.value,
                    "detail": (
                        f"Clause renumbered: AIQS {ref.clause_id} → "
                        f"AIQS {new_id}; update citation"
                    ),
                })
            elif t == TransformationType.CLAUSE_SPLIT:
                new_ids = ", ".join(entry.v2_clause_ids)
                results.append({
                    "wi_id": wi.wi_id,
                    "clause_id": ref.clause_id,
                    "position_bucket": ref.position_bucket.value,
                    "expected_behavior": "edit_required",
                    "transformation_type": t.value,
                    "detail": (
                        f"Clause split into: AIQS {new_ids}; "
                        "update citation to correct sub-clause"
                    ),
                })
            elif t in (
                TransformationType.TERM_REPLACE,
                TransformationType.STRENGTHEN,
                TransformationType.SCOPE_NARROW,
            ):
                payload_str = ", ".join(
                    f"{k}={v}" for k, v in entry.payload.items()
                )
                results.append({
                    "wi_id": wi.wi_id,
                    "clause_id": ref.clause_id,
                    "position_bucket": ref.position_bucket.value,
                    "expected_behavior": "edit_required",
                    "transformation_type": t.value,
                    "detail": (
                        f"Content changed ({t.value}): {payload_str}; "
                        "update WI text to align with v2"
                    ),
                })
            else:
                # clause_insert entries have no v1 IDs so won't match here
                results.append({
                    "wi_id": wi.wi_id,
                    "clause_id": ref.clause_id,
                    "position_bucket": ref.position_bucket.value,
                    "expected_behavior": "edit_required",
                    "transformation_type": t.value,
                    "detail": f"Transformation {t.value} detected; edit required",
                })

    return results


# ---------------------------------------------------------------------------
# scaffold_semantic_changes() — blank template for hand-labeling
# ---------------------------------------------------------------------------


def scaffold_semantic_changes(
    log: TransformationLog,
) -> list[dict]:
    """Emit a scaffold JSON for hand-labeling semantic changes.

    For each semantic log entry, output a template with the description
    and an empty expected_behavior field to be filled manually.
    """
    scaffold: list[dict] = []
    for entry in log.semantic_entries():
        scaffold.append({
            "transformation_type": entry.transformation_type.value,
            "v1_clause_ids": entry.v1_clause_ids,
            "v2_clause_ids": entry.v2_clause_ids,
            "description": entry.payload.get("description", ""),
            "expected_behavior": "",  # fill manually: edit_required | flag_for_review | no_action_required
            "rationale": "",         # fill manually: why this behavior is expected
        })
    return scaffold


# ---------------------------------------------------------------------------
# render_all() — write all artifacts to disk as Markdown / JSON
# ---------------------------------------------------------------------------


def _render_standard_md(std: Standard) -> str:
    """Render a Standard object as Markdown."""
    lines = [f"# {std.title} ({std.version.upper()})", ""]
    for ch in std.chapters:
        lines.append(f"## Chapter {ch.chapter_number}: {ch.title}")
        lines.append("")
        for sec in ch.sections:
            lines.append(
                f"### {ch.chapter_number}.{sec.section_number} {sec.title}"
            )
            lines.append("")
            for cl in sec.clauses:
                lines.append(f"#### {cl.clause_id} {cl.heading}")
                lines.append("")
                lines.append(cl.body)
                lines.append("")
    return "\n".join(lines)


def render_all(
    v1: Standard,
    v2: Standard,
    wis: list[WorkInstruction],
    glossary: list[dict],
    log: TransformationLog,
    expected_edits: list[dict],
    semantic_scaffold: list[dict],
    verbose: bool = False,
) -> None:
    """Write all corpus artifacts to data/."""
    import json as _json

    # Create directories
    for d in [STANDARDS_DIR, WI_DIR, GROUND_TRUTH_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Standards as Markdown
    v1_path = STANDARDS_DIR / "acme_qs_v1.md"
    v2_path = STANDARDS_DIR / "acme_qs_v2.md"
    v1_path.write_text(_render_standard_md(v1), encoding="utf-8")
    v2_path.write_text(_render_standard_md(v2), encoding="utf-8")
    if verbose:
        print(f"  wrote {v1_path}")
        print(f"  wrote {v2_path}")

    # Work Instructions as Markdown
    for wi in wis:
        wi_path = WI_DIR / f"{wi.wi_id}.md"
        wi_path.write_text(wi.body_markdown, encoding="utf-8")
    if verbose:
        print(f"  wrote {len(wis)} WIs to {WI_DIR}/")

    # Glossary
    glossary_path = DATA_DIR / "glossary.json"
    with open(glossary_path, "w") as f:
        _json.dump(glossary, f, indent=2)
    if verbose:
        print(f"  wrote {glossary_path}")

    # Transformation log
    log_path = DATA_DIR / "transformation_log.json"
    with open(log_path, "w") as f:
        _json.dump(log.model_dump(), f, indent=2)
    if verbose:
        print(f"  wrote {log_path}")

    # Ground truth — expected edits
    edits_path = GROUND_TRUTH_DIR / "expected_edits.json"
    with open(edits_path, "w") as f:
        _json.dump(expected_edits, f, indent=2)
    if verbose:
        print(f"  wrote {edits_path}")

    # Ground truth — semantic scaffold
    scaffold_path = GROUND_TRUTH_DIR / "semantic_changes_scaffold.json"
    with open(scaffold_path, "w") as f:
        _json.dump(semantic_scaffold, f, indent=2)
    if verbose:
        print(f"  wrote {scaffold_path}")

    # WI metadata (structured references, for pipeline evaluation)
    wi_meta = []
    for wi in wis:
        wi_meta.append({
            "wi_id": wi.wi_id,
            "title": wi.title,
            "length_bucket": wi.length_bucket.value,
            "topic": wi.topic,
            "references": [
                {
                    "clause_id": r.clause_id,
                    "position_bucket": r.position_bucket.value,
                    "is_edit_requiring": r.is_edit_requiring,
                }
                for r in wi.references
            ],
        })
    meta_path = DATA_DIR / "wi_metadata.json"
    with open(meta_path, "w") as f:
        _json.dump(wi_meta, f, indent=2)
    if verbose:
        print(f"  wrote {meta_path}")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Generate the synthetic corpus for GenAI WI Updater."
    )
    parser.add_argument(
        "--build-v1",
        action="store_true",
        help="Call the LLM to build v1 (25 API calls). "
        "Without this, expects --from-v1.",
    )
    parser.add_argument(
        "--from-v1",
        default=None,
        help="Path to a pre-built v1 JSON. Skips build_v1().",
    )
    parser.add_argument(
        "--skip-wis",
        action="store_true",
        help="Skip WI generation (30 LLM calls). Useful for testing.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to dump v1 as JSON (only with --build-v1).",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    glossary = build_glossary()

    # --- Step 1: v1 standard ---
    if args.from_v1:
        print(f"Loading v1 from {args.from_v1}")
        with open(args.from_v1) as f:
            v1 = Standard.model_validate(_json.load(f))
    elif args.build_v1:
        print("Building v1 Standard — 25 LLM calls")
        rng_build = random.Random(SEED)
        v1 = build_v1(glossary, rng_build, verbose=True)
        out_path = args.out or "/tmp/v1_standard.json"
        with open(out_path, "w") as f:
            _json.dump(v1.model_dump(), f, indent=2)
        print(f"  v1 saved to {out_path}")
    else:
        parser.error("Provide --build-v1 or --from-v1 <path>.")

    print(f"  v1: {len(v1.all_clauses())} clauses")

    # --- Step 2: transform to v2 (pure Python, deterministic) ---
    print("\nTransforming v1 → v2 (mechanical)...")
    v2, log = transform_to_v2(v1, glossary, rng)
    print(f"  v2: {len(v2.all_clauses())} clauses, {len(log.entries)} log entries")

    # --- Step 3: semantic changes ---
    print("\nApplying semantic changes...")
    apply_semantic_changes(v2, log)
    print(f"  log entries after semantic: {len(log.entries)}")
    print(f"  v2 clauses after merges: {len(v2.all_clauses())}")

    # --- Step 4: Work Instructions ---
    wis: list[WorkInstruction] = []
    if not args.skip_wis:
        print("\nBuilding 30 Work Instructions — 30 LLM calls")
        wis = build_wis(v1, log, glossary, rng, n=30, verbose=True)
    else:
        print("\n  --skip-wis: skipping WI generation")

    # --- Step 5: expected edits ---
    print("\nComputing expected edits...")
    expected_edits = compute_expected_edits(wis, log)
    n_edit = sum(1 for e in expected_edits if e["expected_behavior"] == "edit_required")
    n_flag = sum(1 for e in expected_edits if e["expected_behavior"] == "flag_for_review")
    n_noop = sum(1 for e in expected_edits if e["expected_behavior"] == "no_action_required")
    print(f"  {len(expected_edits)} total: {n_edit} edit_required, "
          f"{n_flag} flag_for_review, {n_noop} no_action_required")

    # --- Step 6: semantic scaffold ---
    print("\nBuilding semantic changes scaffold...")
    semantic_scaffold = scaffold_semantic_changes(log)
    print(f"  {len(semantic_scaffold)} entries for hand-labeling")

    # --- Step 7: render all to disk ---
    print("\nRendering all artifacts to data/...")
    render_all(v1, v2, wis, glossary, log, expected_edits, semantic_scaffold,
               verbose=True)

    print("\n✓ Corpus generation complete.")


if __name__ == "__main__":
    main()
