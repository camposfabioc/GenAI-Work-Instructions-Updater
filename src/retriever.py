"""ChromaDB retriever for v2 standard clauses.

Indexes all v2 clauses at startup (heading + body as document text).
Ephemeral — rebuilt every run; 104 clauses embed in seconds.
Uses ChromaDB's default embedding model (all-MiniLM-L6-v2).

Day 3 additions:
- ``ChangeMap`` — structural diff between v1 and v2 clause IDs
- ``retrieve_by_id`` — direct clause lookup by ID
- ``retrieve_narrowed`` — semantic search restricted to a subset of clause IDs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import chromadb

from schemas import Clause, Standard

# Body-similarity threshold for distinguishing genuine content updates from
# ID collisions caused by cascading renumbers.  Calibrated on the synthetic
# corpus: all real content updates score ≥ 0.63, all collisions score ≤ 0.12.
_ID_COLLISION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# ChangeMap — structural diff v1 ↔ v2
# ---------------------------------------------------------------------------


@dataclass
class ChangeMap:
    """Structural + content comparison of clauses between v1 and v2.

    Computed once before processing any WI.  The pipeline uses this to
    decide retrieval strategy per citation — not the transformation_log,
    which is reserved for evaluation only.

    Categories:
    - ``changed``: clause ID present in v1 AND v2, body text differs but
      is sufficiently similar (SequenceMatcher ratio ≥ 0.5) — a genuine
      content update (strengthening, term replacement, etc.).
    - ``id_collision``: clause ID present in v1 AND v2, but body text is
      very different (ratio < 0.5) — a different clause now occupies this
      ID due to cascading renumbers.  Treated like ``v1_only``: the
      pipeline must search all of v2 to find where the original clause
      went.
    - ``unchanged``: clause ID present in v1 AND v2 with identical body.
      No action needed — skip entirely (no LLM call).
    - ``v1_only``: clause ID present in v1 but NOT v2 (deprecated, renumbered
      away, or split — pipeline does not know which).
    - ``v2_only``: clause ID present in v2 but NOT v1 (newly inserted, or
      target of a renumber/split — these are the fallback search space when
      a v1_only citation needs to find its v2 equivalent).
    """

    changed: set[str] = field(default_factory=set)
    id_collision: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)
    v1_only: set[str] = field(default_factory=set)
    v2_only: set[str] = field(default_factory=set)


def build_change_map(v1: Standard, v2: Standard) -> ChangeMap:
    """Compare v1 and v2 clause IDs and bodies to build a structural diff.

    For shared IDs with differing body text, uses ``SequenceMatcher`` to
    distinguish genuine content updates (ratio ≥ 0.5) from ID collisions
    caused by cascading renumbers (ratio < 0.5).
    """
    v1_ids = {c.clause_id for c in v1.all_clauses()}
    v2_ids = {c.clause_id for c in v2.all_clauses()}

    shared_ids = v1_ids & v2_ids
    changed: set[str] = set()
    id_collision: set[str] = set()
    unchanged: set[str] = set()

    for cid in shared_ids:
        v1_clause = v1.clause_by_id(cid)
        v2_clause = v2.clause_by_id(cid)
        v1_body = v1_clause.body.strip()
        v2_body = v2_clause.body.strip()
        if v1_body == v2_body:
            unchanged.add(cid)
        elif SequenceMatcher(None, v1_body, v2_body).ratio() >= _ID_COLLISION_THRESHOLD:
            changed.add(cid)
        else:
            id_collision.add(cid)

    return ChangeMap(
        changed=changed,
        id_collision=id_collision,
        unchanged=unchanged,
        v1_only=v1_ids - v2_ids,
        v2_only=v2_ids - v1_ids,
    )


# ---------------------------------------------------------------------------
# ClauseRetriever
# ---------------------------------------------------------------------------


class ClauseRetriever:
    """Semantic search over v2 standard clauses with direct-lookup support."""

    def __init__(self, v2: Standard) -> None:
        self._client = chromadb.Client()  # ephemeral, in-memory
        self._collection = self._client.create_collection(
            name="v2_clauses",
            metadata={"hnsw:space": "cosine"},
        )
        self._clause_map: dict[str, Clause] = {}

        docs: list[str] = []
        ids: list[str] = []

        for clause in v2.all_clauses():
            doc_text = f"{clause.clause_id} {clause.heading}\n{clause.body}"
            docs.append(doc_text)
            ids.append(clause.clause_id)
            self._clause_map[clause.clause_id] = clause

        self._collection.add(documents=docs, ids=ids)

    # -- Direct lookup (O(1)) -----------------------------------------------

    def retrieve_by_id(self, clause_id: str) -> Clause | None:
        """Return a v2 clause by exact ID, or None if not found."""
        return self._clause_map.get(clause_id)

    # -- Full semantic search -----------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Clause, float]]:
        """Return top-k v2 clauses ranked by cosine similarity.

        Parameters
        ----------
        query : str
            Text to search against (typically a WI chunk).
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[Clause, float]]
            Pairs of (clause, similarity_score) ordered by relevance.
            Score is cosine similarity in [0, 1] — higher is more similar.
        """
        results = self._collection.query(query_texts=[query], n_results=k)

        pairs: list[tuple[Clause, float]] = []
        for clause_id, distance in zip(
            results["ids"][0], results["distances"][0]
        ):
            # ChromaDB returns cosine *distance* (1 - similarity)
            similarity = 1.0 - distance
            pairs.append((self._clause_map[clause_id], similarity))

        return pairs

    # -- Narrowed semantic search (v2-only subset) --------------------------

    def retrieve_narrowed(
        self,
        query: str,
        allowed_ids: set[str],
        k: int = 3,
    ) -> list[tuple[Clause, float]]:
        """Semantic search restricted to a subset of clause IDs.

        Used when a v1 citation is not found in v2 (v1_only).  Instead of
        searching all ~104 clauses, we search only among ``allowed_ids``
        (typically the ``v2_only`` set, ~17 clauses) to find the most likely
        renumber/split target.

        Parameters
        ----------
        query : str
            Text to search against (typically the v1 clause body).
        allowed_ids : set[str]
            Only return results whose clause_id is in this set.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[Clause, float]]
            Pairs of (clause, similarity_score), filtered and ordered.
        """
        if not allowed_ids:
            return []

        # Fetch more than k so filtering doesn't leave us empty
        fetch_k = min(len(self._clause_map), max(k * 3, 10))
        results = self._collection.query(query_texts=[query], n_results=fetch_k)

        pairs: list[tuple[Clause, float]] = []
        for clause_id, distance in zip(
            results["ids"][0], results["distances"][0]
        ):
            if clause_id not in allowed_ids:
                continue
            similarity = 1.0 - distance
            pairs.append((self._clause_map[clause_id], similarity))
            if len(pairs) >= k:
                break

        return pairs
