# Improved Pipeline Evaluation

Run: 2026-05-03T09:13:28.566594+00:00 | Model: gpt-4o-mini | Seed: 20260423 | n_wis: 30

---

## Gate metric

### M0 — Retrieval recall@k

**Not applicable to the improved pipeline.** M0 measures pure embedding-search retrieval (chunk text as query → top-k v2 clauses). The improved pipeline uses hybrid retrieval: direct ID lookup for `changed` clauses, full semantic search for `id_collision` cases, and narrowed search for `v1_only` cases. The numbers below reflect baseline-style retrieval and are shown for comparability only. Retrieval effectiveness of the improved pipeline is captured indirectly by M3 (position recall) and M4 (rule consistency).

How often the retriever returns the correct v2 clause in the top-k results, given a chunk of WI text. This is the *gate*: if retrieval is broken, no downstream metric is meaningful. Failed retrievals propagate as hallucinations in the improved pipeline.

| k | Recall |
|---|---|
| 1 | 0.06 |
| 3 | 0.06 |
| 5 | 0.06 ❌ (target ≥0.85) |

**n_queries:** 54 (edit-requiring references; deprecated clauses excluded)

---

## Failure-mode metrics

### M1 — Reference hallucination

Fraction of proposed edits citing a clause ID that does not exist in v2. Captures the PoC failure where the LLM invents AIQS X.Y.Z citations or keeps v1-only IDs that were renumbered/deprecated. Lower is better.

| Rate | n_proposals | n_hallucinated |
|---|---|---|
| 2.5% | 79 | 2 |

### M2 — Substantive hallucination

Fraction of proposed `new_text` strings that an LLM judge (gpt-4o, temp=0) rules as *not entailed* by the cited v2 clause. Captures the PoC failure where the pipeline writes plausible-sounding text disconnected from the standard. Hallucinated references (M1) and citation-only changes (where `old_text` and `new_text` differ only by their AIQS reference number) are excluded from the denominator — those are covered by M1 and M4. Lower is better.

_Skipped (--skip-m2)_

### M3 — Lost-in-middle (action recall by WI position)

Recall of edit/flag actions broken down by where the citation sits in the WI (top/middle/bottom thirds, normalized by section index). Captures the PoC failure where edits in the middle of long documents are missed disproportionately. Looking for: a dip in the middle bucket compared to top and bottom.

| Bucket | Recall | n |
|---|---|---|
| Top    | 77.8%    | 9 |
| Middle | 75.0% | 20 |
| Bottom | 62.1% | 29 |

**Gap (max(top,bot) − middle):** 0.03 — high gap = strong lost-in-middle effect.

### M4 — Rule consistency (action rate by transformation type)

For each v1→v2 transformation type, fraction of expected actions the pipeline proposed (edit OR flag — any non-silent response). Captures the PoC failure where the same rule is applied inconsistently across different WIs. Looking for: low std_dev across types, no type below ~0.7.

| Type | Action rate | n |
|---|---|---|
| clause_split | 100.0% | 4 |
| scope_narrow | 100.0% | 4 |
| strengthen | 100.0% | 9 |
| term_replace | 84.6% | 13 |
| deprecate | 75.0% | 4 |
| renumber | 37.5% | 24 |

**Std dev:** 0.22 | **Min rate:** 37.5%

### M5 — Terminology compliance

Three sub-metrics, computed by string matching against the glossary:

- **Preservation:** when an unchanged equipment ID appears in `old_text`, the same ID appears verbatim in `new_text`. Pipeline must not mutate opaque tokens.
- **Migration:** when a superseded equipment ID appears in `old_text`, the pipeline replaces it with the correct `superseded_by` ID in `new_text`.
- **Expansion accuracy:** when the pipeline expands an abbreviation in `new_text` (e.g. `ORR (...)`), the expansion matches the canonical one from the glossary.

| Sub-metric | Rate | n |
|---|---|---|
| Preservation | 96.7% | 30 |
| Migration | 100.0% | 34 |
| Expansion accuracy | 27.3% | 22 |

### Deprecated handling

For the v1 clauses deprecated in v2 and cited in WIs, the schema-correct action is `flag` (no concrete edit can be auto-proposed). Tracks whether the pipeline distinguishes 'edit' from 'flag' use cases.

| Outcome | Count |
|---|---|
| Correctly flagged | 3 |
| Incorrectly proposed edit | 0 |
| Missed (no action) | 1 |
| **Total deprecated** | **4** |

---

## Ops

| Metric | Value |
|---|---|
| Total cost | $0.0221 |
| Total tokens | 121,610 |
| Total LLM calls | 80 |
| Avg latency per call | 2.15s |
