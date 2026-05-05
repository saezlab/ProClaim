# Upstream FIRE, SAFE, and OpenScholar baseline integration

## Summary

- Replaced the local FIRE-inspired baseline loop with a thin wrapper around `experiments/fire/eval/fire/verify_atomic_claim.py`.
- Replaced the local SAFE-inspired baseline loop with a thin wrapper around `experiments/long-form-factuality/eval/safe/rate_atomic_fact.py`.
- Kept OpenScholar as a subprocess-backed wrapper, but corrected its default repo root to `experiments/OpenScholar`.

## Design choices

- FIRE preserves the upstream prompting and control flow. SAFE preserves the upstream control flow, but its final-verdict prompt is patched to reuse the repo's shared SUPPORT / REFUTE / UNCERTAIN definitions.
- Instead, both upstream search hooks are patched to call the repo's existing shared web helper `baselines.react_baseline._do_search`, matching the requested "use do search instead" direction.
- FIRE and SAFE are now web-only in the baseline runner. The earlier local reimplementation supported `search_backend=s2`, but the upstream repos do not expose an equivalent Semantic Scholar mode.
- The runner no longer passes `search_backend` into FIRE or SAFE and no longer creates a fixed `safe/web/...` namespace in results paths; both baselines now write to `results/baselines/{fire,safe}/<model>/...`.
- SAFE is integrated through the upstream `rate_atomic_fact` stage rather than the full prompt-response pipeline because our datasets already provide atomic claims rather than long-form responses that need fact decomposition.

## Runtime integration details

- The runner still uses the shared `LLMBackend`, but a small adapter layer now presents the interface expected by each upstream repo.
- The adapters record every upstream prompt so the existing per-claim logs continue to include prompts.
- Small import-time stubs are installed for upstream `common.modeling`, `common.utils`, and `common.shared_config` so the verification modules can load without their optional original client stack or CLI-only utility dependencies.

## Validation

- Static diagnostics (`get_errors`) passed on the touched baseline files.
- Constructor-level runtime validation succeeded for `FIREBaseline`, `SAFEBaseline`, and `OpenScholarBaseline` under the workspace venv and the same `sys.path` shape used by `experiments/run_baselines_datasets.py`.

## Known divergences from the original FIRE framework

The following differences exist between running `FIREBaseline` in this pipeline and running the original FIRE repo end-to-end. They are listed roughly by expected impact on evaluation metrics.

### 1. Temperature (high impact)
The original `common/modeling.py` defaults to `temperature=0.5`. `LLMBackend` defaults to `temperature=0.0`. At 0.0 the model is greedy-deterministic: it follows the same reasoning path and issues the same search queries on every run. FIRE was designed for stochastic exploration, so the local pipeline is more consistent but less thorough — it may exit the iterative loop earlier.

### 2. Search backend (high impact)
The original FIRE calls the Serper Google Search API (`query_serper.SerperAPI`, `num_searches=3`). The patched `call_search` delegates to `react_baseline._do_search`, which uses DuckDuckGo. Because the LLM reasons solely over retrieved snippets, a different search engine directly changes the evidence and therefore the verdict. Note also that `FIREBaseline.num_search_results` (settable via `--top-k`) is **not** forwarded into `_patched_search`; the patched function always issues 5 results regardless of the flag.

### 3. JSON extraction regex — `re.DOTALL` (moderate impact)
`common/utils.extract_json_from_output` uses `re.search(r'{.*}', output)` without `re.DOTALL`, so `.` does not match newlines. The local stub in `upstream_adapters.import_upstream_module` uses `re.DOTALL`. Modern LLMs frequently emit multi-line JSON objects; without the flag the original FIRE fails to parse these and burns a retry. The local pipeline parses them on the first attempt, so the 10-retry budget is consumed much more slowly and the forced-final-answer fallback is reached less often.

### 4. Max tokens (minor impact)
Original default: `max_tokens=2048`. Local CLI default: `4096` (via `--max-tokens`). FIRE's reasoning rarely hits 2 k tokens in practice, so this seldom changes a verdict.

### What stays identical
- System prompt text (`"You are a fact-checking agent responsible for verifying the accuracy of claims."`)
- All upstream prompts (`_FINAL_ANSWER_OR_NEXT_SEARCH_FORMAT`, `_MUST_HAVE_FINAL_ANSWER_FORMAT`)
- Entire `verify_atomic_claim` control flow, including `max_steps`, `max_retries`, `max_tolerance=2`, `diverse_prompt=False`
- `generate()` return contract — `FireModelAdapter.generate` returns `(text, {"input_tokens": …, "output_tokens": …})`, matching exactly what the upstream loop expects