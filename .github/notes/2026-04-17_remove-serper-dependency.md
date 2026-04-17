# Remove Serper API Dependency — 2026-04-17

## Branch

`exp`

## Summary

Removed the paid Serper API (`google.serper.dev`) as a web search backend from all three baseline modules that used it. Web search now exclusively uses DuckDuckGo via the free `ddgs` package. This eliminates the `SERPER_API_KEY` requirement and simplifies the search stack to a single, zero-cost backend for all web-search-enabled baselines (retrieval, react, fire).

## Modified Files

| File | Changes |
|------|---------|
| `experiments/baselines/retrieval_baseline.py` | Removed `_serper_search()`, `_SERPER_URL` constant, `SERPER_API_KEY` env check in `_web_search()`. Dropped `os` and `requests` imports. Updated module docstring. |
| `experiments/baselines/react_baseline.py` | Removed `_serper_search()`, `_SERPER_URL` constant, `SERPER_API_KEY` env check in `_do_search()`. Dropped `os` and `requests` imports. Updated module docstring. Wired `num_search_results` default to `5` (matching `--top-k`) instead of hardcoded `3`. |
| `experiments/baselines/fire_baseline.py` | Removed `_serper_search()`, `_SERPER_URL` constant, `self._serper_key` attribute. Renamed `_google_search()` → `_web_search()` and simplified the call site in `_step()`. Dropped `os` and `requests` imports. Updated module docstring. |
| `experiments/run_baselines_datasets.py` | Updated `--search-backend` CLI help text: `Serper/DuckDuckGo` → `DuckDuckGo`. Passed `num_search_results=args.top_k` to `ReActBaseline` so search result count is driven by `--top-k` config. |
| `experiments/README.md` | Updated `--search-backend` flag description, prerequisites for Retrieval/ReAct/FIRE to reflect DuckDuckGo-only web search. |

## Key Design Decisions

- **DuckDuckGo as sole web backend**: Serper provided marginally higher-quality Google results but required a paid API key, creating a barrier to reproducibility. DuckDuckGo via `ddgs` is free, requires no API key, and is already a project dependency.
- **No fallback chain**: Previously, code checked for `SERPER_API_KEY` and fell back to DuckDuckGo. With only one backend, the dispatch logic is removed entirely, reducing code complexity.
- **FIRE function rename**: `_google_search()` was renamed to `_web_search()` to accurately reflect that it uses DuckDuckGo, not Google. The previous name was a legacy artifact from the original FIRE paper implementation.
- **ReAct `num_search_results` unified with `--top-k`**: Previously `_NUM_SEARCH_RESULTS` was hardcoded to `3` and not wired to the CLI. Now the factory passes `args.top_k` as `num_search_results`, and the default matches `--top-k` (5).
