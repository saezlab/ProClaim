# SIGNOR Eval Test — Sample Run Issue Analysis (2026-03-25)

**Branch:** `exp/init`

## Summary

Deep analysis of a single sample run from `results/verification/signor_eval_test/` (FES→BCR, SIGNOR-45343, forward orientation, rep 1). The run reaches the correct verdict (SUPPORT @ 0.75 confidence) but exposes six systemic problems in the evidence verification pipeline: subagent thinking-token waste, extreme paper attrition, a non-functional sufficiency classifier, workspace path mismatches, subagent model quality gaps, and orchestrator API gaps forcing manual workarounds.

## Run Context

| Field | Value |
|-------|-------|
| Claim | FES directly inhibits BCR (either through inhibition or destabilization) |
| Edge | SIGNOR-45343, FES→BCR, flip=False, rep=1 |
| Ground truth | SUPPORTED |
| Verdict | SUPPORT @ 0.75 confidence (correct) |
| Iterations | 4 (02:12–03:21 UTC, ~70 min) |
| Papers retrieved | 69 total across all iterations |
| Papers surviving filter | 3 (same three every iteration: 8955135, 10706130, 15302586) |

## Issues Identified

### Issue 1: Subagent Thinking-Token Pathology on Irrelevant Papers (P0 — cost/latency)

When Qwen 3.5 9B receives an irrelevant paper (PMID 23316053 — an Abl kinase minireview that only mentions c-Fes in passing), it enters a catastrophic reasoning loop: **~29K characters of thinking tokens** that ultimately produce **empty output** (0 chars). The model deliberates endlessly about whether `[]` is an acceptable response, whether "Abl is distinct from c-Fes" constitutes a NEUTRAL fact, and loops on the same reasoning ~30 times before emitting nothing.

Qwen3-8B exhibits the identical pattern (29,096 chars thinking → empty output). Claude Haiku 4.5 on the same paper returns `[]` in 14 characters with zero thinking overhead.

**Root cause:** The `extract_facts` prompt in `subagents.py` says *"Output ONLY a JSON array of fact objects"* but gives no explicit guidance for the irrelevant-paper case. The model enters an underdetermined state.

**Evidence:**
- `debug_logs/2026-03-24T21-54-15.701_pmid_23316053.json` — Qwen 3.5 9B: 28,968 chars thinking → 0 chars output
- `debug_logs/2026-03-25T11-54-13.856_pmid_23316053.json` — Qwen3-8B: 29,096 chars thinking → 0 chars output
- `debug_logs/2026-03-25T12-52-20.620_pmid_23316053_claude.json` — Claude Haiku 4.5: 0 chars thinking → 14 chars output (`[]`)

**Proposed fix:** Add an explicit escape clause to the `extract_facts()` prompt: *"If the paper does not contain any facts relevant to the claim or its subclaims, output an empty JSON array `[]` immediately. Do not deliberate."*

**Files:** `src/proclaim/verification/subagents.py` L59–105 (prompt construction)

---

### Issue 2: Massive Paper Attrition — 96% Waste Rate (P0 — cost)

The `trace.json` shows `filter_papers_by_stance()` removes 79–88% of papers every iteration, always leaving the same 3 papers:

| Iteration | Papers Before | Papers After | Removed | % Removed |
|-----------|--------------|-------------|---------|-----------|
| 1 | 14 | 3 | 11 | 79% |
| 2 | 16 | 3 | 13 | 81% |
| 3 | 14 | 3 | 11 | 79% |
| 4 | 25 | 3 | 22 | 88% |

57 unique papers are removed across iterations. Each one went through full-text retrieval → NLP feature computation → fact extraction → filtering — all wasted.

**Cascade:**
1. PubMed search retrieves papers that mention gene names but don't discuss the specific FES–BCR relationship (BCR-Abl reviews, general kinase papers, Rho GTPase papers).
2. Fact extraction correctly labels them NEUTRAL or extracts nothing.
3. `filter_papers_by_stance()` removes all NEUTRAL-only papers.
4. Gap search retrieves more tangentially related papers → repeat.

**Root cause:** No pre-extraction relevance filter. Every retrieved paper pays the full processing cost before being assessed for relevance.

**Proposed fix:** Add a lightweight relevance pre-screen (e.g., co-occurrence of both gene names in abstract, or a fast LLM binary-relevance check) *before* expensive fact extraction and feature computation.

**Files:** `src/proclaim/verification/evidence_api.py` L1557–1610 (`filter_papers_by_stance`), L783–906 (`extract_and_add_facts*`)

---

### Issue 3: Sufficiency Classifier Plateau at 0.59 (P0 — cost/accuracy)

The MLP sufficiency classifier returns **confidence 0.59 on every single iteration**. The notebook documents flat confidence across all 4 checks. The agent eventually overrides this by emitting a verdict at 0.75 confidence based on its own evidence reading.

**Root cause:** This is the RC-2 pattern from the prior analysis. After filtering removes all but 3 papers, the feature aggregator always processes the same papers with the same features. Newly added papers are removed before scoring. The z-score-normalized feature vector produces a constant logit → constant probability → the classifier is a no-op.

The `min_papers_per_iteration=3` override doesn't even trigger here because the MLP never says "sufficient" — it's stuck outputting the same insufficient probability.

**Impact:** 3 extra iterations (~50 min of wall time) spent trying to satisfy a classifier that cannot be satisfied. The orchestrator correctly identifies sufficient evidence through its own analysis but has no API to bypass the classifier.

**Proposed fix:** (a) Detect feature stagnation — if MLP confidence changes < 0.01 for 2 consecutive iterations, allow early verdict. (b) Expose a `force_verdict` parameter in `emit_verdict()` that skips sufficiency requirements. (c) Fix the underlying RC-2 feature-zero problem per prior analysis.

**Files:** `src/proclaim/verification/evidence_api.py` L1476–1551 (`check_sufficiency`)

---

### Issue 4: Workspace Path Mismatch (P1 — debuggability)

The `evidence_state.json` in the test folder is **empty** (427 bytes, freshly initialized template with `papers: {}`, `facts: []`, `iteration: 0`). The notebook's `setup_kernel()` call points to a different absolute path:

```
/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct/results/signor_eval/SIGNOR-45343/...
```

Note `rain` vs `ail` in the base path. The actual state was persisted to the original eval workspace. Only `verdict.json`, `trace.json`, and `debug_logs/` were copied to this test folder.

**Impact:** Post-hoc debugging requires access to two separate directory trees. The evidence_state (which contains the full paper/fact inventory) is not co-located with the analysis artifacts.

**Proposed fix:** Either (a) make workspace paths relative to the notebook location, or (b) have `emit_verdict()` save a full state snapshot alongside the verdict in the same directory.

---

### Issue 5: Subagent Quality Gap — Qwen vs Claude (P1 — accuracy)

Comparing the two subagent models on PMID 8955135 (the primary evidence paper):

| Metric | Qwen 3.5 9B | Claude Haiku 4.5 |
|--------|-------------|------------------|
| Thinking overhead | ~22K chars | 0 |
| Facts extracted | 5 | 7 |
| Subclaim coverage | 3 of 4 | 4 of 4 |
| Confidence calibration | 1.0 on all facts | 0.90–0.95 (more realistic) |
| Handles unaddressed subclaim | Silently omits | Explicit NEUTRAL fact |
| Output format | Clean JSON | JSON in markdown fences |

Key differences:
- Claude extracts a NEUTRAL fact for the destabilization subclaim (*"The paper does not describe FES destabilizing BCR protein"*), providing explicit negative coverage that reduces gap queries for that subclaim. Qwen omits it, leaving a coverage hole.
- Claude's confidence values (0.90–0.95) are better calibrated than Qwen's blanket 1.0, which inflates the confidence signal.
- Claude wraps output in ` ```json ``` ` fences — requires `_parse_facts_response()` to strip markdown formatting, a known fragility.

**Files:** `src/proclaim/verification/subagents.py` (prompt + `_parse_facts_response`)

---

### Issue 6: Orchestrator Improvisation — API Surface Gaps (P1 — maintainability)

The notebook reveals the Claude orchestrator progressively abandoning the structured API and improvising:

1. **Multi-query search** (cells 7–9): The agent calls `search_pubmed()` 8+ times with manually crafted queries because `search_pubmed_llm()` generates only one query. The API lacks a multi-query search function.

2. **Manual retry of failed extractions** (cell 13): The agent manually retries 7 failed PMIDs by calling `extract_facts()` + `add_facts_from_dicts()` directly, bypassing `extract_and_add_facts()`. The batch API doesn't report which papers failed or offer retry.

3. **Manual verdict override** (cell 50): The agent emits a verdict with manually chosen confidence (0.75) and hand-written reasoning, completely bypassing the MLP confidence signal that plateaued at 0.59.

**Proposed API additions:**
- `search_pubmed_multi(claim, state, llm, num_queries=5)` — LLM generates multiple diverse queries in one call
- `extract_and_add_facts_batch()` return value should include failed PMIDs with error reasons, plus a built-in retry mechanism
- `emit_verdict()` should accept an optional `override_confidence` flag that bypasses the sufficiency threshold

**Files:** `src/proclaim/verification/evidence_api.py`, `src/proclaim/verification/evidence_programming.py` (system prompt workflow steps)

---

## Combined Priority Matrix

| ID | Issue | Priority | Impact | Category |
|----|-------|----------|--------|----------|
| 1 | Thinking-token loops on irrelevant papers | P0 | ~29K wasted tokens/paper × 11–22 papers/iter | Cost |
| 2 | 96% paper waste rate (no pre-filter) | P0 | 4–6× wasted retrieval + NLP cost | Cost |
| 3 | Sufficiency classifier plateau | P0 | 3 extra iterations (~50 min) per run | Cost + Latency |
| 4 | Workspace path mismatch | P1 | Debug difficulty | Debuggability |
| 5 | Qwen vs Claude subagent quality gap | P1 | Missing subclaim coverage, poor calibration | Accuracy |
| 6 | API surface gaps → orchestrator improvisation | P1 | Code duplication in notebooks, fragile workarounds | Maintainability |

## Files Analyzed

| File | Purpose |
|------|---------|
| `results/verification/signor_eval_test/evidence_report.ipynb` | 52-cell orchestrator notebook (4 iterations) |
| `results/verification/signor_eval_test/workspace/verdict.json` | Final verdict: SUPPORT @ 0.75 |
| `results/verification/signor_eval_test/workspace/trace.json` | 4 filter operations showing paper attrition |
| `results/verification/signor_eval_test/workspace/evidence_state.json` | Empty (state saved to different path) |
| `results/verification/signor_eval_test/workspace/debug_logs/*.json` | 5 subagent debug logs (Qwen 3.5 9B, Qwen3-8B, Claude Haiku 4.5) |
| `src/proclaim/verification/evidence_api.py` | Evidence API (search, extract, filter, sufficiency, verdict) |
| `src/proclaim/verification/subagents.py` | LLM subagent functions (fact extraction, gap identification) |
| `src/proclaim/verification/evidence_programming.py` | Claude Agent SDK orchestrator + system prompt |
| `src/proclaim/verification/llm_factory.py` | LLM client factory (timeout, retry logic) |

---

## Planned Code Changes (Issue 1 fix: Disable Qwen Thinking Mode)

### Problem recap

vLLM launches Qwen3 models with `--reasoning-parser qwen3` (in `scripts/vllm_node_setup.sh` L88), which enables server-side thinking mode. However, neither of the two `make_llm()` call sites for subagents passes `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to disable it. The `make_llm()` docstring at `llm_factory.py` L67 documents this exact kwarg but it was never wired into actual calls.

With thinking enabled, the Qwen model allocates part of its `max_tokens` budget (default 8,000) to `<think>` tokens. The vLLM reasoning parser routes these to `delta.reasoning_content` (which our streaming handler correctly ignores at L113), but the thinking still consumes GPU time and can starve the content output when the model enters circular reasoning on ambiguous inputs.

### Qwen3 technical report findings (https://qwen.ai/blog?id=qwen3)

**Reasoning loop issue:** Not mentioned. The blog describes hybrid thinking/non-thinking modes purely as a feature. The pathology we observe (model consuming its entire `max_tokens` on `<think>` tokens for simple extraction prompts) is an emergent behavior on structured-extraction tasks that don't benefit from extended reasoning.

**Performance impact of disabling thinking:** Minimal for our workload. Qwen3's 4-stage post-training pipeline includes **Stage 3: "thinking mode fusion"** — the model is explicitly fine-tuned to produce high-quality direct responses when thinking is off. This is not "thinking mode with the switch flipped" but a separately trained capability. The blog states performance curves are *"scalable and smooth"* with thinking budget, confirming no cliff when disabling. Our subagent tasks (fact extraction, synthesis, gap identification) are structured-extraction, not multi-step reasoning — thinking adds cost without benefit.

**vLLM parser note:** The blog's own vLLM example uses `--reasoning-parser deepseek_r1` for Qwen3, while our `vllm_node_setup.sh` uses `--reasoning-parser qwen3`. Both parsers work but it's worth noting the official recommendation differs. Not a blocker for the `enable_thinking: False` fix since that operates at the chat-template level, not the parser level.

### What to change

#### 1. `src/proclaim/verification/config.py` — `make_subagent_llm()` (~L298)

**Current code:**
```python
def make_subagent_llm(self):
    from proclaim.verification.llm_factory import make_llm
    return make_llm(
        base_url=self.subagent_base_url,
        api_key=self.api_key,
        model=self.subagent_model,
    )
```

**Planned change:** Add `extra_body` with `enable_thinking: False`:
```python
def make_subagent_llm(self):
    from proclaim.verification.llm_factory import make_llm
    return make_llm(
        base_url=self.subagent_base_url,
        api_key=self.api_key,
        model=self.subagent_model,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
```

**Why here:** This is the primary factory used by `VerificationSettings` when the system is run via config / CLI. All subagent calls (fact extraction, synthesis, conflict detection, gap identification) go through this callable.

#### 2. `src/proclaim/verification/evidence_api.py` — `setup_kernel()` (~L1822)

**Current code:**
```python
llm = make_llm(base_url=base_url, api_key=api_key, model=model)
```

**Planned change:**
```python
llm = make_llm(
    base_url=base_url,
    api_key=api_key,
    model=model,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

**Why here:** `setup_kernel()` is the bootstrap path used when the orchestrator calls `nb_execute` to initialize the Jupyter kernel. It constructs its own `make_llm()` call from environment variables rather than going through `config.py`. Both paths must be patched.

### Safety considerations

- **Non-Qwen models:** The `extra_body` field is passed through to the API request body. Non-Qwen backends (GLM, Llama, Mistral) will ignore `chat_template_kwargs` since their chat templates don't define an `enable_thinking` parameter. Verified by checking `llm_factory.py` L105 and L142 — `extra_body` is forwarded unconditionally to `client.chat.completions.create()`.
- **No other `make_llm()` call sites:** Grep confirms only two production call sites: `config.py` L311 and `evidence_api.py` L1822. The third hit in `llm_factory.py` L77 is a docstring example.
- **Reversibility:** If thinking mode is later needed for a specific task, it can be re-enabled per-call by passing `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` to `make_llm()` at that call site.
