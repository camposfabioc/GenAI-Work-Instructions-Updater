"""Diagnostic: inspect what the retriever returns for one specific
expected-edit entry. Helps debug M0 = 0 results.

Usage:
    python scripts/debug_retrieval.py WI-001 3.4.4
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import json
import re

from chunker import chunk_wi
from retriever import ClauseRetriever
from schemas import Standard, TransformationLog


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/debug_retrieval.py <WI-ID> <v1_clause_id>")
        sys.exit(1)

    wi_id, v1_id = sys.argv[1], sys.argv[2]
    data = _REPO_ROOT / "data"

    v2 = Standard.model_validate(
        json.loads((data / "standards" / "acme_qs_v2.json").read_text())
    )
    log = TransformationLog.model_validate(
        json.loads((data / "transformation_log.json").read_text())
    )
    wi_md = (data / "work_instructions" / f"{wi_id}.md").read_text()

    # 1. v2 targets for this v1 ID
    v2_targets = []
    for entry in log.entries:
        if entry.is_semantic:
            continue
        if v1_id in entry.v1_clause_ids:
            v2_targets = list(entry.v2_clause_ids)
            print(f"Found in transformation log: {entry.transformation_type}")
            print(f"  v1 IDs: {entry.v1_clause_ids}")
            print(f"  v2 IDs: {entry.v2_clause_ids}")
            break
    if not v2_targets:
        print(f"No log entry — assuming unchanged. v2_targets = [{v1_id}]")
        v2_targets = [v1_id]

    print(f"\nExpected v2 targets: {v2_targets}")

    # 2. Find chunk containing AIQS reference
    chunks = chunk_wi(wi_md)
    print(f"\nWI has {len(chunks)} chunks")
    pattern = re.compile(rf"\bAIQS\s+{re.escape(v1_id)}\b")
    matched_chunk = None
    for ch in chunks:
        if pattern.search(ch.text):
            matched_chunk = ch
            break

    if matched_chunk is None:
        print(f"ERROR: No chunk contains 'AIQS {v1_id}'")
        print("\nFirst 200 chars of each chunk's text:")
        for i, ch in enumerate(chunks):
            print(f"  [{i}] {ch.text[:200].replace(chr(10), ' / ')}")
        return

    print(f"\nMatched chunk (idx={matched_chunk.chunk_index}, bucket={matched_chunk.position_bucket}):")
    print("-" * 70)
    print(matched_chunk.text)
    print("-" * 70)

    # 3. Retrieve top-5
    retriever = ClauseRetriever(v2)
    results = retriever.retrieve(matched_chunk.text, k=5)

    print("\nTop-5 retrieved clauses:")
    for i, (clause, score) in enumerate(results, 1):
        marker = "  ← MATCH" if clause.clause_id in v2_targets else ""
        print(f"  {i}. [{score:.3f}] {clause.clause_id} — {clause.heading}{marker}")


if __name__ == "__main__":
    main()
