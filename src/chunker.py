"""Chunk Work Instruction Markdown by section headings.

Splitting strategy:
- Every ``##`` or ``###`` heading starts a new chunk.
- The ``#`` title line (WI name) is discarded — it is metadata, not content.
- If a chunk exceeds 800 tokens, it is split at the nearest paragraph boundary.
- No overlap between chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

TOKEN_CEILING = 800
_HEADING_RE = re.compile(r"^(#{2,3})\s+", re.MULTILINE)
_TITLE_RE = re.compile(r"^#\s+", re.MULTILINE)

_enc = tiktoken.encoding_for_model("gpt-4o-mini")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Chunk:
    """One chunk of a Work Instruction."""

    text: str
    heading: str
    chunk_index: int
    total_chunks: int  # filled after all chunks are built

    @property
    def position_ratio(self) -> float:
        """Normalized position in [0, 1].  0 = first chunk, 1 = last."""
        if self.total_chunks <= 1:
            return 0.0
        return self.chunk_index / (self.total_chunks - 1)

    @property
    def position_bucket(self) -> str:
        r = self.position_ratio
        if r < 0.33:
            return "top"
        if r < 0.67:
            return "middle"
        return "bottom"


def _split_by_headings(md: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) pairs split on ## and ### headings.

    The ``#`` title line is skipped entirely.
    """
    # Remove title line(s)
    md = _TITLE_RE.sub("", md, count=1).lstrip("\n")

    parts: list[tuple[str, str]] = []
    positions = [m.start() for m in _HEADING_RE.finditer(md)]

    if not positions:
        # No headings at all — return whole text as single chunk
        return [("(no heading)", md.strip())]

    # Text before the first heading (rare, but handle it)
    if positions[0] > 0:
        preamble = md[: positions[0]].strip()
        if preamble:
            parts.append(("(preamble)", preamble))

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(md)
        block = md[start:end].strip()

        # Extract heading text (first line)
        first_nl = block.find("\n")
        if first_nl == -1:
            heading = block
            body = ""
        else:
            heading = block[:first_nl].strip()
            body = block[first_nl + 1 :].strip()

        full_text = block  # keep heading + body together
        parts.append((heading, full_text))

    return parts


def _split_long_chunk(text: str, heading: str) -> list[tuple[str, str]]:
    """Split a chunk that exceeds TOKEN_CEILING at paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    sub_chunks: list[tuple[str, str]] = []
    buffer = ""

    for para in paragraphs:
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if _token_count(candidate) > TOKEN_CEILING and buffer:
            sub_chunks.append((heading, buffer.strip()))
            buffer = para
        else:
            buffer = candidate

    if buffer.strip():
        sub_chunks.append((heading, buffer.strip()))

    return sub_chunks


def chunk_wi(markdown: str) -> list[Chunk]:
    """Split a WI Markdown document into chunks.

    Parameters
    ----------
    markdown : str
        Full text of a Work Instruction Markdown file.

    Returns
    -------
    list[Chunk]
        Ordered list of chunks with position metadata.
    """
    raw_parts = _split_by_headings(markdown)

    # Enforce token ceiling
    final_parts: list[tuple[str, str]] = []
    for heading, text in raw_parts:
        if _token_count(text) > TOKEN_CEILING:
            final_parts.extend(_split_long_chunk(text, heading))
        else:
            final_parts.append((heading, text))

    total = len(final_parts)
    return [
        Chunk(text=text, heading=heading, chunk_index=i, total_chunks=total)
        for i, (heading, text) in enumerate(final_parts)
    ]
