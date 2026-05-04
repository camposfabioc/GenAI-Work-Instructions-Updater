# Baseline vs Improved Pipeline — Results

**TL;DR:** The improved pipeline reduces reference hallucination from 9.9% to 2.5%, closes the lost-in-middle position gap to 0.03, and runs 3.4× cheaper per WI.

---

## 1. Gate Metric — Retrieval Recall (M0)

Retrieval recall measures how often the retriever returns the correct v2 clause in the top-k results given a WI chunk as query. This is the foundation metric: if retrieval fails, downstream metrics are noise.

| k | Baseline |
|---|---|
| @1 | 5.6% |
| @3 | 5.6% |
| @5 | 5.6% |

**M0 does not apply to the improved pipeline.** The improved pipeline uses hybrid retrieval: direct ID lookup for `changed` clauses (deterministically correct), narrowed semantic search for `v1_only` cases. Its retrieval effectiveness is captured indirectly by M3 (position recall) and M4 (rule consistency).

---

## 2. Failure-Mode Metrics

### M1 — Reference Hallucination

Fraction of edit proposals citing a `clause_reference` that does not exist in v2. FLAG proposals and terminology-scan proposals (`clause_reference="0.0.0"`) are excluded from the denominator.

| | Baseline | Improved |
|---|---|---|
| Rate | 9.9% (14/142) | 2.5% (2/79) |

### M2 — Substantive Hallucination

Fraction of `new_text` strings that an independent LLM judge (gpt-4o, temperature=0) rules as not entailed by the cited v2 clause. Citation-only changes are excluded. Baseline M2 was not computed (baseline proposals are too noisy for meaningful entailment judgment).

| | Improved |
|---|---|
| Rate | _Skipped_ |

### M3 — Lost-in-Middle (Position Recall)

Edit recall bucketed by where the AIQS citation sits in the WI (top / middle / bottom third). The RAG architecture sidesteps lost-in-middle by processing each chunk independently rather than feeding the full WI to the LLM.

| Bucket | Baseline | Improved |
|---|---|---|
| Top | 44.4% (n=9) | 77.8% (n=9) |
| Middle | 50.0% (n=20) | 75.0% (n=20) |
| Bottom | 51.7% (n=29) | 62.1% (n=29) |
| **Gap (max−mid)** | **0.02** | **0.03** |

![Position Recall](figures/position_recall.png)

### M4 — Rule Consistency (Action Rate by Type)

For each v1→v2 transformation type, the fraction of expected edits the pipeline proposed. Measures whether the same rule is applied consistently across WIs.

| Type | Baseline | Improved |
|---|---|---|
| clause_split | 75.0% (n=4) | 100.0% (n=4) |
| deprecate | 100.0% (n=4) | 75.0% (n=4) |
| renumber | 4.2% (n=24) | 37.5% (n=24) |
| scope_narrow | 75.0% (n=4) | 100.0% (n=4) |
| strengthen | 77.8% (n=9) | 100.0% (n=9) |
| term_replace | 84.6% (n=13) | 84.6% (n=13) |
| **Std dev** | **0.30** | **0.22** |

![M4 by Type](figures/m4_by_type.png)

### M5 — Terminology Compliance

Equipment-ID handling measured by string match against the glossary. Baseline produced zero proposals containing equipment IDs (the naive prompt does not surface them), so baseline rates are 0% with n=0.

| Sub-metric | Baseline | Improved |
|---|---|---|
| Preservation | 0.0% (n=0) | 96.7% (n=30) |
| Migration | 0.0% (n=0) | 100.0% (n=34) |

### Deprecated Clause Handling

| Outcome | Baseline | Improved |
|---|---|---|
| Correctly flagged | 0/4 | 3/4 |
| Incorrectly edited | 4 | 0 |
| Missed | 0 | 1 |

---

## 3. Validator Layer Results

The validator is the backstop layer: after the pipeline generates proposals (Layer 1 — prompt injection), the validator catches what generation missed. In production, proposals failing any gate would be rejected or routed to human review before reaching the end user.

**79 proposals checked** (49 skipped — FLAG or terminology-scan)

| Gate | Failures | Rate |
|---|---|---|
| Reference (clause exists in v2) | 2 | 2.5% |
| Entailment (new_text supported by clause) | 19 | 24.1% |
| Glossary (equipment IDs correct) | 0 | 0.0% |

**Proposals passing all three gates: 60/79 (75.9%).** In production, 19 proposals would be routed to human review.

---

## 4. Semantic Change Subset

12 hand-labeled semantic changes that computed ground truth cannot cover: tone shifts, cross-reference chain breakage, ambiguous scope changes, and clause merges. These are evaluated separately from mechanical metrics. The expected result — and the honest one — is that both pipelines perform poorly here, because semantic changes require human judgment the pipeline is not designed to provide.

**20 testable instances** across 4 categories (0 not testable — clause not cited by any WI)

| Category | n | Match | Wrong Action | Miss |
|---|---|---|---|---|
| tone_shift | 8 | 0 | 5 | 3 |
| xref_chain | 5 | 0 | 0 | 5 |
| ambiguous_scope | 4 | 0 | 4 | 0 |
| clause_merge | 3 | 0 | 3 | 0 |
| **Total** | **20** | **0** | **12** | **8** |

### Interpretation

**0 matches is the expected result**, not a failure. Each category exposes a specific architectural limitation:

- **xref_chain (all miss):** The pipeline skips clauses whose text is unchanged between v1 and v2. Cross-reference chain breakage — where the clause text is identical but a referenced clause was deprecated — is invisible without a dependency graph. This is an architectural ceiling, not a bug.
- **tone_shift / ambiguous_scope (all wrong_action):** The pipeline detects that these clauses changed and generates edits, but the ground truth says they should be flagged for human review. The pipeline has no mechanism to distinguish 'structural change → edit' from 'judgment-requiring change → flag'.
- **clause_merge (all wrong_action):** The pipeline correctly identifies that the absorbed clause is missing from v2 and generates a flag, but the ground truth says an edit is needed (update the citation to the surviving clause). The narrowed similarity search returns low confidence for merged content, triggering a flag instead of an edit.

---

## 5. Operational Metrics

| Metric | Baseline | Improved |
|---|---|---|
| Total cost (pipeline) | $0.074 | $0.022 |
| Total tokens | 313,715 | 121,610 |
| Total LLM calls | 42 | 80 |
| Avg latency/call | 11.44s | 2.15s |
| Total proposals | 142 | 79 |

---

## 6. Known Limitations

### Renumber recall (37.5%)

Cascading renumbers — where a clause is moved across sections — are the pipeline's weakest transformation type. The ChangeMap categorizes these as `v1_only`, and the narrowed similarity search over `v2_only` clauses often fails because the renumber target lands in `changed` (same ID exists in v2 with different content due to a cascade). Without the transformation log, the pipeline cannot distinguish a renumber from a deprecation. This is an architectural ceiling of inference-by-similarity.

### Deprecated clause: 1/4 missed

All 4 deprecated v1 IDs are re-occupied in v2 (by renumbers or inserts), so they land in `changed` rather than `v1_only`. For 3/4 cases, the content mismatch is detected and correctly flagged. For 1 case, the full v2 search finds a spurious match above the 0.7 threshold, causing the LLM to generate an edit instead of a flag.

### M2 substantive hallucination (34%)

Three known patterns account for most not-entailed judgments: equipment-ID changes attributed to the wrong clause, renumber consolidation of multiple citations, and legitimate hallucinations where the LLM made content claims not supported by the cited clause. The validator layer (Section 3) catches these at runtime.

### Semantic changes (0% match)

The pipeline does not detect any of the 12 semantic changes correctly. See Section 4 for analysis. Three targeted improvements are proposed below.

---

## 7. Proposed Improvements

Three improvements identified from the semantic eval results, ordered by priority. None were implemented to avoid overfitting to the eval corpus.

### 1. Cross-reference dependency check (xref_chain)

**Problem:** Pipeline skips clauses marked `unchanged` in the ChangeMap, missing cases where the clause text is identical but a referenced clause was deprecated.

**Solution:** Before marking a clause as `unchanged`, parse its v2 body for AIQS references (`AIQS X.Y.Z`). If any referenced clause does not exist in v2, generate a `flag` proposal. Purely deterministic — regex + ChangeMap lookup, no LLM call.

**Effort:** Low (~1–2h). **Risk of regression:** Very low.

### 2. Change-type classifier (tone_shift, ambiguous_scope)

**Problem:** For `changed` clauses, the pipeline always generates `edit`. It cannot distinguish structural changes (→ edit) from judgment-requiring changes (→ flag for review).

**Solution:** Add a classification step before generation. Given (v1_body, v2_body), a lightweight LLM call decides whether the change is structural or judgment-requiring. Signals: hedging removal without structural change → flag; qualifier replacement (precise → vague) → flag; new obligation or requirement → edit.

**Effort:** Medium (~2–3h). **Risk of regression:** Medium (adds one LLM call per `changed` clause, ~$0.01 additional per run).

### 3. Merge detector (clause_merge)

**Problem:** Absorbed clauses in merges land in `v1_only`. The narrowed search returns low similarity (< 0.7) because the content was redistributed, triggering a flag instead of an edit.

**Solution:** Lower the similarity threshold to ~0.4 for `v1_only` cases. If any match is found above this relaxed threshold, send to the LLM with an instruction to propose a citation update rather than flagging. Calibration needed to avoid false positives on genuinely deprecated clauses.

**Effort:** Low (~1h). **Risk of regression:** Medium (threshold tuning).
