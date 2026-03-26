# SIGNOR Evaluation Results Analysis — 2026-03-24

**Branch:** `exp/init`

## Summary

Comprehensive analysis of the SIGNOR dataset evaluation results (`results/signor_eval_results.csv`) produced by the RLM evidence verification agent. Covers 14 SIGNOR edges, each evaluated 3× in forward and flipped orientations (80 total runs). Overall accuracy is **63.7%** (51/80). Analysis spans two dimensions: (A) dataset/claim-level failure modes, and (B) architecture-level inefficiencies driving cost and runtime.

---

## 1. Architecture Overview

- **Orchestrator**: Claude Sonnet 4-6 via Claude Agent SDK ($3/M input, $15/M output, 1.25× cache write, 0.1× cache read)
- **Subagent**: Qwen 3.5 9B on local vLLM server (OpenAI-compatible API, called from inside Jupyter kernel)
- **Architecture pattern**: RLM (Recursive Language Model) — Claude writes Python code → `nb_execute` runs it in persistent Jupyter kernel → results fed back as conversation context
- **MCP Tools**: notebook-tools server (nb_init, nb_execute, nb_markdown, nb_render_*, nb_read_output, nb_save)
- **Sufficiency classifier**: Pre-trained MLP on NLP+metadata features (spaCy, SBERT similarity, NLI cross-encoder, publication year, impact factor, citations, h-index), threshold=0.5
- **Full-text retrieval**: 4-layer chain: PMC Open Access → EuropePMC → INDRA → Unpaywall+PDF, max 50K chars per paper
- **Evidence pipeline per iteration**: `search_pubmed_llm` → `extract_and_add_facts_batch` → `populate_paper_features` → `check_sufficiency` → `emit_verdict`

---

## 2. Key Metrics

| Metric | Value |
|--------|-------|
| Total runs | 80 |
| Unique edges | 14 |
| Overall accuracy | 63.7% (51/80) |
| Forward (flip=False) accuracy | 58.5% (24/41) |
| Flipped (flip=True) accuracy | 69.2% (27/39) |
| Inter-rep consistency (SPLIT rate) | 29.6% (8/27 combos) |
| Mean confidence (correct) | 0.846 |
| Mean confidence (incorrect) | 0.711 |
| Total cost | $88.34 (~$1.10/run) |
| Mean tokens/run | ~1.48M input, ~15K output |

### Accuracy by Ground Truth Label

| Label | Accuracy |
|-------|----------|
| SUPPORTED (10 edges) | 70.0% (42/60) |
| WRONG (3 edges) | 50.0% (7/14) |
| UNCERTAIN (1 edge) | 33.3% (2/6) |

### Confusion Matrix (Mapped Verdict vs Expected Label)

|  | SUPPORTED | UNCERTAIN | WRONG |
|---|-----------|-----------|-------|
| **Predicted SUPPORTED** | 21 | 1 | 8 |
| **Predicted UNCERTAIN** | 7 | 2 | 2 |
| **Predicted WRONG** | 8 | 3 | 28 |

---

## 3. Cost & Token Analysis

### Cost Breakdown by Component

| Component | Share |
|-----------|-------|
| Cache write tokens | 42.1% |
| Cache read tokens | 36.9% |
| Output tokens | 20.9% |
| Raw input tokens | ~0.0% |

- **Cache hit ratio**: 90.6% average
- **Incorrect runs cost more**: $1.23 avg vs $1.03 for correct runs (more iterations → more confusion)
- **High cost variance**: Same edge, different reps can vary 2–7× (e.g., MTCP1→AKT2: $0.39 vs $2.73)

### Expensive vs Cheap Run Comparison

| Metric | Expensive (MAPK8IP2→MAPK9 flip=T rep=3) | Cheap (CSK→SRC flip=F rep=1) |
|--------|------------------------------------------|-------------------------------|
| Cost | $4.91 | $0.26 |
| nb_execute calls | 1,185 | 105 |
| Runtime | ~2h 20min | ~10min |
| Behaviour | Agent desperately searches for JIP2/MAPK8IP2 papers, manually reads abstracts, checks relevance one-by-one | Clean pipeline: search → extract → sufficient → verdict |

### Root Causes of Runaway Cost & Poor Performance

Three root causes were identified through code inspection (2026-03-25). The previously hypothesised `min_papers_per_iteration=3` override is a **secondary** effect — the three issues below are the primary drivers.

#### RC-1: Subagent LLM calls hang without timeout (cost & runtime)

The OpenAI client in `llm_factory.py` L91 is created **without any explicit timeout**:
```python
client = OpenAI(base_url=base_url, api_key=api_key)  # no timeout kwarg
```
The default OpenAI SDK timeout is 600s (10 min). The retry logic (3 attempts, exponential backoff at L155–182) means a single hung vLLM endpoint blocks for **up to 30 minutes per subagent call**. All 5 subagent call sites in `subagents.py` call `llm(prompt)` with no try/except or per-call timeout:
- `extract_facts()` — L161
- `synthesize_subclaim()` — L242
- `detect_conflicts()` — L286
- `identify_gaps()` — L381 (called from `check_sufficiency()` on every insufficient iteration)
- `formulate_gap_queries()` — L495

Since `extract_and_add_facts_batch()` runs 8 threads in parallel via `ThreadPoolExecutor`, a single hung request blocks that thread slot indefinitely while the orchestrator waits.

**Fix:** Pass `timeout=httpx.Timeout(120, connect=10)` to the `OpenAI()` constructor. Add a per-call hard timeout (e.g. `signal.alarm` or `concurrent.futures` wrapper) around each `llm()` invocation in subagents.

**Files:** `src/pkevolve/verification/llm_factory.py` L91, `src/pkevolve/verification/subagents.py` L161/242/286/381/495

#### RC-2: Sufficiency classifier falls back to deterministic default output (accuracy)

The MLP classifier in `mlp_predict()` (`scripts/sufficiency_classifier/test_mlp_classifier.py` L80–87) fills **0.0 for every missing feature**:
```python
val = flat.get(fname)
if val is None or not isinstance(val, (int, float)):
    vec.append(0.0)  # ← silent zero-fill
```

Features go missing through a multi-step chain:
1. `populate_paper_features()` (`evidence_api.py` L967) catches all exceptions per-paper and logs them but **does not propagate** — the paper simply has `nlp=None` / `metadata=None`.
2. In `check_sufficiency()` (`evidence_api.py` L1396–1402), papers with `None` metadata/nlp produce **empty dicts `{}`** in the feature list.
3. The `FeatureAggregator.aggregate_all()` receives mostly-empty dicts and returns zeros/empty for all aggregated features.
4. `flatten_features()` finds no matching keys → `mlp_predict` builds an **all-zeros vector**.
5. After z-score normalization `(0 - mean) / scale`, this produces a **deterministic constant logit** regardless of actual evidence.

The sufficiency decision is therefore meaningless whenever `populate_paper_features` fails (e.g. NLI model OOM, spaCy model not loaded, metadata API timeout). The classifier always outputs the same probability, and the `min_papers_per_iteration` override then becomes the de facto decision maker — explaining the retry loops.

**Fix:** (a) Add a guard in `check_sufficiency()`: if >50% of papers lack NLP features, skip MLP and return `insufficient` with an explicit diagnostic. (b) Add a warning/metric when `mlp_predict` receives an all-zeros input vector. (c) Consider making `populate_paper_features` failures fatal or at least propagating a feature-coverage ratio.

**Files:** `src/pkevolve/verification/evidence_api.py` L1370–1420, `scripts/sufficiency_classifier/test_mlp_classifier.py` L80–87

#### RC-3: Sequential full-text retrieval is slow with unbounded layers (runtime)

`fetch_full_text()` in `full_text.py` L514–540 runs **4 layers sequentially**. If each layer fails, the cumulative worst-case time per paper is:

| Layer | Timeout | Worst case |
|-------|---------|------------|
| PMC (elink + efetch) | 30s + 60s | 90s |
| Europe PMC (search + fetch) | 15s + 60s | 75s |
| INDRA (`get_full_text()`) | **no explicit timeout** | ∞ |
| DOI resolve + Unpaywall + PDF parse | 15s + 15s + 60s | 90s |

When all layers fail (common for non-OA papers), a single paper takes **4–6+ minutes**. The INDRA layer delegates to `indra.literature.get_full_text()` which has **no timeout at all** — a hung INDRA endpoint blocks the thread indefinitely.

Even though `extract_and_add_facts_batch()` parallelises with 8 workers, each worker runs the full sequential chain per paper. For a batch of 10 papers where most are non-OA, the wall time is dominated by the slowest paper (potentially infinite via INDRA).

**Fix:** (a) Add `timeout=30` to the INDRA layer. (b) Add an overall per-paper timeout of 120s wrapping the entire `fetch_full_text()` call. (c) Consider parallelising the 4 layers (race them concurrently, take the first success). (d) Reduce `max_chars` in `full_text.py` from 50K to 20K to match the 16K truncation in `extract_facts()` (`subagents.py` L157) — 68% of fetched text is currently discarded.

**Files:** `src/pkevolve/verification/full_text.py` L68–540, `src/pkevolve/verification/subagents.py` L157

---

## 4. Dataset-Level Failure Modes

### 4.1 Loss of Mechanistic Nuance in Claim Construction (P0)

`construct_signor_claim()` collapses diverse SIGNOR `EFFECT` values into binary "activates"/"inhibits" templates. Fails on edges where the interaction mechanism is ambiguous:

- **CHUK→NFKBIA** (SUPPORTED, "inhibits"): Agent finds IKKβ is the primary kinase; over-qualifies CHUK's role → UNCERTAIN/REFUTE
- **SRC→CTTN** (SUPPORTED, "inhibits"): SRC phosphorylation inhibits F-actin crosslinking but activates migration → REFUTE
- **BMI1→H2AX** (SUPPORTED, "activates"): BMI1 ubiquitinates H2AX (not classic activation) → UNCERTAIN/REFUTE

**Proposed fix:** Include the SIGNOR `MECHANISM` column in claim construction; decompose into mechanism-specific subclaims.

### 4.2 REFUTE Bias on Flipped UNCERTAIN Edges (P0)

**MAPK8IP2→MAPK9** (UNCERTAIN): When flipped to "inhibits", agent unanimously says REFUTE 3/3, but expected label is still UNCERTAIN. The scaffold protein has context-dependent effects.

**Proposed fix:** Add conflict-aware verdict logic in `emit_verdict()` — when SUPPORT facts >30% AND REFUTE facts >30%, prefer UNCERTAIN.

### 4.3 Directional Ambiguity in Post-Translational Modifications (P1)

**FES→BCR**: Forward (inhibits) → SUPPORT 3/3 ✓. Flipped (activates) → SUPPORT 3/3 ✗. Agent interprets phosphorylation as "activation" in flipped context, but phosphorylation actually suppresses BCR's kinase activity.

**Proposed fix:** Counter-argument search step after preliminary verdict; fact extraction should capture net functional outcome.

### 4.4 Potential Ground Truth Quality Issues (P2)

- **PRKDC→AKT1** (labeled WRONG): Agent says SUPPORT 2/2 with high confidence. Multiple in-vitro studies show DNA-PKcs phosphorylates AKT at S473.
- **PPP1CA→BRCA1** (labeled WRONG): Agent uncertain/split. PP1α dephosphorylation of BRCA1 is interpretable as activation or inhibition.

**Proposed action:** Flag high-confidence disagreements (>0.80) for manual ground truth review.

### 4.5 Poor Confidence Calibration (P1)

Incorrect predictions have mean confidence 0.711, barely below 0.846 for correct ones. Agent is overconfident on errors (e.g., FES→BCR flipped: SUPPORT at 0.80–0.92, all wrong).

**Proposed fix:** Confidence penalties for mixed fact distributions; self-consistency check across forward/flipped pairs.

---

## 5. Architecture-Level Fixes (Verified Root Causes)

The three root causes below were verified by code inspection on 2026-03-25. They supersede the earlier hypotheses in the original analysis. The `min_papers_per_iteration=3` override is a **secondary symptom** — it only matters because the MLP classifier is outputting garbage (RC-2), causing the override to become the de-facto decision maker.

### 5.1 RC-1 Fix: Add Timeout to Subagent LLM Calls (P0 — runtime)

**Root cause:** `OpenAI()` client created without timeout (`llm_factory.py` L91). Default SDK timeout is 600s × 3 retries = up to 30 min per call. All 5 subagent functions in `subagents.py` call `llm(prompt)` with no wrapping timeout or exception handling.

**Fix:**
1. `llm_factory.py` L91: `OpenAI(base_url=..., api_key=..., timeout=httpx.Timeout(120, connect=10))`
2. Each subagent call site: wrap in try/except with a hard timeout (e.g. `concurrent.futures.ThreadPoolExecutor` with `future.result(timeout=150)`).
3. The retry loop (L155–182) should catch `httpx.TimeoutException` explicitly rather than generic `Exception`.

**Files:** `src/pkevolve/verification/llm_factory.py` L91, L155–182; `src/pkevolve/verification/subagents.py` L161, L242, L286, L381, L495
**Projected impact:** Eliminates unbounded hangs. Worst-case subagent call capped at ~2.5 min instead of 30 min.

### 5.2 RC-2 Fix: Prevent Classifier Zero-Vector Fallback (P0 — accuracy & cost)

**Root cause:** When `populate_paper_features()` fails silently (exceptions caught per-paper), papers have `nlp=None`/`metadata=None`. The feature aggregator receives empty dicts, `flatten_features()` finds no keys, and `mlp_predict()` fills an all-zeros vector. After z-score normalization, this produces a **deterministic constant probability** regardless of evidence — making the sufficiency classifier a no-op.

**Fix:**
1. `evidence_api.py` `check_sufficiency()` (~L1390): Before calling `mlp_predict`, count papers with non-None `nlp` and `metadata`. If coverage < 50%, skip MLP entirely and return `insufficient` with diagnostic `"feature_coverage_too_low"`.
2. `test_mlp_classifier.py` `mlp_predict()` L80: Add a check — if all features are 0.0, log a warning and return `(0.5, 0.0)` with a flag indicating default output.
3. `evidence_api.py` `populate_paper_features()`: Consider making NLP feature failures propagate (or at minimum, return a coverage metric in the status string).

**Files:** `src/pkevolve/verification/evidence_api.py` L1370–1420, L967–1100; `scripts/sufficiency_classifier/test_mlp_classifier.py` L80–87
**Projected impact:** Fixes meaningless sufficiency decisions → fewer wasted iterations → directly reduces cost on expensive runs. Also improves accuracy by not prematurely terminating when features happen to produce a high-confidence zero-vector output.

### 5.3 RC-3 Fix: Add Timeouts and Parallelise Full-Text Retrieval (P0 — runtime)

**Root cause:** `fetch_full_text()` (`full_text.py` L514–540) runs 4 layers **sequentially**. The INDRA layer (L351–390) has **no timeout**. Worst-case per paper: 90s (PMC) + 75s (EuropePMC) + ∞ (INDRA) + 90s (Unpaywall) = unbounded. Non-OA papers routinely take 4–6+ minutes.

**Fix:**
1. `full_text.py` `_fetch_indra()` (~L356): Wrap `get_full_text()` in a 30s timeout (e.g. `concurrent.futures` or `signal.alarm`).
2. `full_text.py` `fetch_full_text()` (~L514): Add overall per-paper timeout of 120s wrapping the entire function.
3. Consider racing layers concurrently (submit all 4 to a `ThreadPoolExecutor`, take first non-None result, cancel the rest).
4. Reduce `max_chars` default from 50,000 to 20,000 — `extract_facts()` in `subagents.py` L157 truncates to 16K anyway, so 68% of fetched text is discarded.

**Files:** `src/pkevolve/verification/full_text.py` L68–540; `src/pkevolve/verification/subagents.py` L157
**Projected impact:** Per-paper retrieval capped at 120s instead of unbounded. Parallel racing could reduce to ~30s for OA papers. Reduced `max_chars` saves bandwidth and INDRA/Unpaywall latency.

### 5.4 Secondary: `min_papers_per_iteration=3` Override (P2 — cost)

**Note:** The `min_papers_per_iteration=3` override in `check_sufficiency()` (evidence_api.py ~L1435) was previously identified as the primary cost driver. In reality it is a **secondary effect**: it only causes retry loops because the MLP classifier is outputting garbage due to RC-2. Once the classifier produces meaningful probabilities, the override will rarely trigger on edges with sufficient literature. It may still warrant relaxation (e.g. `min_papers_per_iteration=1` or removal after iteration 2), but it is not the root cause.

**Files:** `src/pkevolve/verification/evidence_api.py` L1435–1447

---

## 6. Per-Edge Results Detail

### Perfect Edges (6/6 correct across both orientations)
- **GNAS→ADCY1** — canonical Gs-alpha activates adenylyl cyclase
- **HRAS→PIK3CG** — crystal-structure-level evidence for Ras→PI3Kγ
- **CSK→SRC** — textbook negative regulation
- **AURKA→PLK1** — well-characterized kinase-substrate pair
- **MTCP1→AKT2** — TCL1 family co-activator of AKT
- **FES→BCR** (forward only) — correct on forward, fails on flipped

### Consistently Wrong Edges
- **SRC→CTTN** — 0/6 correct (wrong both orientations, unanimously)
- **FES→BCR** (flipped) — 0/3 correct (always says SUPPORT for "activates")

### Split/Inconsistent Edges
- CHUK→NFKBIA forward: [UNCERTAIN, UNCERTAIN, REFUTE]
- BMI1→H2AX forward: [UNCERTAIN, REFUTE, UNCERTAIN]
- BCL2L1→BAD flipped: [SUPPORT, REFUTE, REFUTE]
- GNAI1→HCK forward: [REFUTE, UNCERTAIN, REFUTE]

---

## 7. Combined Priority Action Items

### Verified Root Causes (P0 — fix before next eval run)

| ID | Action | Root Cause | Files | Impact |
|----|--------|------------|-------|--------|
| RC-1 | Add `timeout=httpx.Timeout(120, connect=10)` to OpenAI client; wrap subagent calls with hard timeout | Subagent hangs block threads for up to 30 min | `llm_factory.py` L91; `subagents.py` L161/242/286/381/495 | Eliminates unbounded hangs |
| RC-2 | Guard `check_sufficiency()` against all-zeros feature vector; add feature coverage check before MLP | MLP outputs deterministic constant when features missing | `evidence_api.py` L1370–1420; `test_mlp_classifier.py` L80–87 | Fixes meaningless sufficiency → fewer wasted iterations |
| RC-3 | Add 30s timeout to INDRA layer; add 120s overall per-paper timeout to `fetch_full_text()`; reduce max_chars 50K→20K | Sequential 4-layer retrieval with no INDRA timeout | `full_text.py` L68–540; `subagents.py` L157 | Per-paper capped at 120s instead of unbounded |

### Dataset-Level Improvements (P0–P1)

| Priority | Action | Category | Files | Impact |
|----------|--------|----------|-------|--------|
| P0 | Include SIGNOR `MECHANISM` in claim construction | Dataset | `experiments/run_signor_eval.py` | Accuracy on mechanistic edges |
| P0 | Conflict-aware verdict logic (bilateral evidence → UNCERTAIN) | Dataset | `evidence_api.py` | UNCERTAIN recall |
| P1 | Counter-argument search after preliminary verdict | Dataset | `evidence_api.py` | Directional ambiguity accuracy |
| P1 | Confidence penalty for mixed SUPPORT/REFUTE fact pools | Dataset | `evidence_api.py` | Calibration |

### Secondary Architecture Improvements (P1–P2)

| Priority | Action | Category | Files | Impact |
|----------|--------|----------|-------|--------|
| P1 | Create composite `run_iteration()` API to reduce nb_execute calls | Architecture | `evidence_api.py`, `evidence_programming.py` | 40–60% cost reduction (context growth) |
| P1 | Move relevance filtering to subagent layer | Architecture | `evidence_api.py`, `subagents.py` | 30–70% cost on hard edges |
| P1 | Early stopping for decisive evidence (heuristic pre-check) | Architecture | `evidence_api.py` | 50% cost on easy edges |
| P2 | Relax `min_papers_per_iteration` override (secondary to RC-2) | Architecture | `evidence_api.py` L1435 | Minor once RC-2 is fixed |
| P2 | Race full-text layers concurrently instead of sequentially | Architecture | `full_text.py` | Further runtime reduction |

### Low Priority (P2–P3)

| Priority | Action | Category | Files | Impact |
|----------|--------|----------|-------|--------|
| P2 | Subclaim decomposition for mechanism + outcome | Dataset | Prompt templates | Accuracy |
| P2 | Self-consistency scoring across forward/flipped pairs | Dataset | Analysis scripts | Consistency |
| P3 | Flag high-confidence GT disagreements for review | Dataset | Analysis scripts | GT quality |

---

## 8. Files Analyzed

| File | Purpose |
|------|---------|
| `results/signor_eval_results.csv` | 80-row evaluation results CSV |
| `results/signor_eval/` | Per-edge evidence notebooks and workspaces |
| `experiments/run_signor_eval.py` | Evaluation orchestration script |
| `experiments/signor_eval_config.yaml` | Config (claude-sonnet-4-6 + qwen3.5-9b subagents) |
| `src/pkevolve/verification/evidence_programming.py` | Main orchestrator (Claude Agent SDK loop, system prompt) |
| `src/pkevolve/verification/evidence_api.py` | Evidence API (search, extract, sufficiency, verdict) |
| `src/pkevolve/verification/subagents.py` | LLM subagent functions (fact extraction, gap identification) |
| `src/pkevolve/verification/full_text.py` | 4-layer full-text retrieval chain |
| `src/pkevolve/verification/data_models.py` | Pydantic models (Fact, PaperRecord, SufficiencyResult, etc.) |
| `src/pkevolve/verification/llm_factory.py` | make_llm() callable factory |
| `src/pkevolve/verification/config.py` | VerificationSettings, build_sdk_env(), dual-endpoint routing |
