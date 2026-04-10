# S2 Retrieval Rate-Limit Fix & Prompt Rule Removal — 2026-04-10

**Branch:** `exp`

## Summary

Three issues were identified and resolved during S2 retrieval baseline experiments on SIGNOR. (1) The `_s2_search` function in `s2_retrieval.py` had no inter-request throttling or retry logic, causing silent 429 (rate-limit) failures from Semantic Scholar's API. With anonymous access (~1 req/s limit), roughly half of all search requests returned empty results due to unhandled 429 responses—not because S2 had no papers. The same rate-limit bug existed in the shared `S2Client` used by the evidence programming system. Both were fixed with 1.1s minimum inter-request intervals and exponential back-off retries. (2) Hardcoded "Rules" sections in `prompts.py` biased verdict behaviour (e.g., "prefer REFUTE over UNCERTAIN") and were not applied consistently across baselines. These were removed from both `build_verification_system_prompt()` and `build_verification_system_prompt_no_retrieval()`, along with unused `DECOMPOSITION_PROMPT` and `QUERY_GENERATION_PROMPT` constants. (3) The ReAct baseline was rewritten to use LangGraph's `create_react_agent` with `ChatLiteLLM` and native tool-use API, and FIRE was refactored to accept the shared `LLMBackend` instead of calling litellm directly.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/s2_retrieval.py` | Fixed-k S2 RAG baseline. Now delegates to `S2Client` from `pkevolve.search.semantic_scholar` instead of raw `requests` calls, inheriting rate-limit and retry logic. |
| `experiments/configs/s2_retrieval_config.yaml` | YAML config for S2 retrieval runs: Claude Sonnet 4.6, top_k=5, SIGNOR + ConnectomeDB. |

## Modified Files

| File | Change |
|------|--------|
| `experiments/baselines/shared/prompts.py` | Removed hardcoded "Rules" block from `build_verification_system_prompt()` (4 rules including "prefer REFUTE over UNCERTAIN") and from `build_verification_system_prompt_no_retrieval()` (2 rules including "prefer UNCERTAIN over SUPPORT"). Removed unused `DECOMPOSITION_PROMPT` and `QUERY_GENERATION_PROMPT` constants. |
| `src/pkevolve/search/semantic_scholar.py` | Added 429 retry with exponential back-off to both `_get()` and `_post()` methods. Up to 3 retries with 2s/4s/8s delays. Added `citationCount,url` to default S2 fields. |
| `experiments/baselines/react_baseline.py` | Rewritten from manual litellm tool-call loop to LangGraph `create_react_agent` + `ChatLiteLLM`. Added S2 search backend (`search_backend="s2"`) alongside existing web search. Added DuckDuckGo thread-safety lock. |
| `experiments/baselines/fire_baseline.py` | Refactored to accept shared `LLMBackend` instance instead of calling `litellm.completion()` directly. Constructor signature now takes `llm: LLMBackend` (model/temperature taken from backend). |
| `experiments/run_baselines_datasets.py` | Registered `s2_retrieval` baseline in `build_baseline()` factory. Added `--s2-model`, `--s2-top-k`, `--react-search-backend`, `--temperature` CLI flags. FIRE now constructed with `LLMBackend`. ACE and ReAct now receive `temperature` argument. |
| `experiments/baselines/shared/llm.py` | Minor: added `load_dotenv()` call at import time (already present, confirmed working). |

## Key Design Decisions

- **Rate-limit throttle at the client layer, not the baseline layer.** The `S2Client._rate_limit()` method enforces a minimum 1.1s gap between any two requests. Individual baselines (S2 retrieval, ReAct with S2 backend, evidence programming) inherit this automatically by using `S2Client` rather than raw HTTP calls.

- **Exponential back-off on 429, not just fixed delays.** Retries use 2s → 4s → 8s delays (3 attempts max). This handles burst throttling while keeping total wait bounded at ~14s per query in the worst case.

- **Removed biasing rules from shared prompts.** The "Rules" blocks in `prompts.py` contained opinionated instructions (e.g., "If no passage mentions the entities, prefer REFUTE over UNCERTAIN") that silently biased results toward REFUTE. These were removed so verdict definitions from `LabelConfig` are the sole source of label semantics—consistent across all baselines.

- **S2 retrieval delegates to `S2Client` singleton.** The baseline previously used inline `requests.get()` calls with no rate limiting. Switching to `S2Client` from `pkevolve.search.semantic_scholar` centralises S2 access, rate limiting, and retry logic in one place.

- **Separate output directory for no-rules experiments.** Results from re-runs with the cleaned prompts are saved to `results/baselines_norules/` to preserve the original `results/baselines/` data for comparison.

## Bug Fixes

- **Silent 429 failures in S2 search (critical).** The original `_s2_search()` caught HTTP 429 responses as generic exceptions and returned `[]`, indistinguishable from "no papers exist." This caused 91/101 SIGNOR claims to receive no evidence at query time (depending on request rate), making both Claude and Gemini default to REFUTE for nearly every claim. After the fix, S2 returns 2–5 results for all 101 claims.

- **S2_API_KEY present but not used effectively.** The key was correctly loaded from `.env` via `load_dotenv()`, but without inter-request delays the authenticated tier still triggered 429s under rapid sequential calls. The throttle fixes this regardless of whether an API key is set.

- **Identical metrics across models and top-k values (explained).** All four experiment configurations (Claude/Gemini × top-5/top-10) produced identical metrics (accuracy=0.6238, macro_f1=0.3663) because both models saw the same (mostly irrelevant) S2 evidence and independently converged on REFUTE. With the rate-limit fix applied, re-runs are expected to produce differentiated results.
