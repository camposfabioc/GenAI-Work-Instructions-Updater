"""Standalone utility: parse a Standard Markdown file into the Pydantic
``Standard`` JSON shape and write it next to the input.

Run once after ``data_gen.py`` to materialize the JSON form of v1/v2 standards
without re-running the LLM. The retriever (Day 2) and validators (Day 3) load
the JSON, not the Markdown.

Usage
-----
    python scripts/export_standard_json.py data/standards/acme_qs_v2.md v2
    python scripts/export_standard_json.py data/standards/acme_qs_v1.md v1

Format expected (consistent with data_gen.py output)
----------------------------------------------------
    # Title line                          → discarded
    ## Chapter N: Chapter Title           → Chapter
    ### N.M Section Title                 → Section
    #### N.M.K Clause heading             → Clause heading
    [single paragraph]                    → Clause body
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make src/ importable so we can validate against Standard
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from schemas import Chapter, Clause, Section, Standard  # noqa: E402

_CHAPTER_RE = re.compile(r"^##\s+Chapter\s+(\d+):\s+(.+?)\s*$")
_SECTION_RE = re.compile(r"^###\s+(\d+)\.(\d+)\s+(.+?)\s*$")
_CLAUSE_RE = re.compile(r"^####\s+(\d+\.\d+\.\d+)\s+(.+?)\s*$")


def parse_standard(md: str, version: str) -> Standard:
    """Parse Markdown text into a Standard object."""
    lines = md.splitlines()

    chapters: list[Chapter] = []
    current_chapter: Chapter | None = None
    current_section: Section | None = None
    current_clause_id: str | None = None
    current_clause_heading: str | None = None
    current_body_lines: list[str] = []

    def flush_clause() -> None:
        nonlocal current_clause_id, current_clause_heading, current_body_lines
        if current_clause_id is None:
            return
        body = " ".join(line.strip() for line in current_body_lines).strip()
        if not body:
            raise ValueError(
                f"Clause {current_clause_id} has empty body"
            )
        clause = Clause(
            clause_id=current_clause_id,
            heading=current_clause_heading,
            body=body,
        )
        current_section.clauses.append(clause)
        current_clause_id = None
        current_clause_heading = None
        current_body_lines = []

    for line in lines:
        if (m := _CHAPTER_RE.match(line)):
            flush_clause()
            current_chapter = Chapter(
                chapter_number=int(m.group(1)),
                title=m.group(2),
                sections=[],
            )
            chapters.append(current_chapter)
            current_section = None
            continue

        if (m := _SECTION_RE.match(line)):
            flush_clause()
            if current_chapter is None:
                raise ValueError(f"Section before any chapter: {line!r}")
            chapter_num = int(m.group(1))
            section_num = int(m.group(2))
            if chapter_num != current_chapter.chapter_number:
                raise ValueError(
                    f"Section {chapter_num}.{section_num} does not match "
                    f"chapter {current_chapter.chapter_number}"
                )
            current_section = Section(
                section_number=section_num,
                title=m.group(3),
                clauses=[],
            )
            current_chapter.sections.append(current_section)
            continue

        if (m := _CLAUSE_RE.match(line)):
            flush_clause()
            if current_section is None:
                raise ValueError(f"Clause before any section: {line!r}")
            current_clause_id = m.group(1)
            current_clause_heading = m.group(2)
            current_body_lines = []
            continue

        # Top-level title (#) and other markers — skip
        if line.startswith("#"):
            continue

        # Body content
        if current_clause_id is not None and line.strip():
            current_body_lines.append(line)

    flush_clause()

    return Standard(version=version, chapters=chapters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path", type=Path, help="Path to the standard .md file")
    ap.add_argument("version", choices=["v1", "v2"], help="Standard version tag")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (defaults to same dir, .json extension)",
    )
    args = ap.parse_args()

    md = args.md_path.read_text(encoding="utf-8")
    standard = parse_standard(md, args.version)

    out_path = args.out or args.md_path.with_suffix(".json")
    out_path.write_text(
        json.dumps(standard.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    n_clauses = len(standard.all_clauses())
    print(
        f"Wrote {out_path} — {len(standard.chapters)} chapters, "
        f"{n_clauses} clauses"
    )


if __name__ == "__main__":
    main()
