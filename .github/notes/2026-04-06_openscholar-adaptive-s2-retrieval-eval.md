# OpenScholar Adaptive S2 Retrieval Evaluation — 2026-04-06

**Branch:** `exp`

## Summary

Implemented the OpenScholar RAG baseline (`experiments/baselines/open_scholar_baseline.py`)
and wired it into the shared evaluation harness.  The baseline routes each claim
through the OpenScholar subprocess pipeline, reframing it as a yes/no literature
question and requesting a structured JSON verdict (`claim_verdict_question` task).

A full evaluation of all 111 SIGNOR claims was run with zero-shot inference,
S2 adaptive retrieval (`--ss_retriever --feedback`), CE reranking
(`--ranking_ce --reranker OpenScholar/OpenScholar_Reranker`), `top_n=10`, and
`claude-sonnet-4-6` via the Anthropic API.  An explicit `oracle_context` flag
was added to prevent ground-truth CSV evidence from being silently injected,
separating oracle-leakage and pure-retrieval evaluation modes.

A root-cause analysis identified three failure modes that collapsed adaptive
performance well below the non-adaptive (oracle-injected) baseline: feedback-loop
format corruption, domain-mismatched S2 keyword queries, and 429 rate errors
from the free-tier S2 API.  Per-run YAML config files were also introduced to
avoid having to repeat long CLI flag sequences.  The evaluation metrics harness
was extended with weighted FPR/FNR in addition to the existing macro variant.

## New Files

| File | Purpose |
|---|---|
| `experiments/baselines/open_scholar_baseline.py` | Full OpenScholar RAG baseline. Wraps `OpenScholar/run.py` as a subprocess. Uses `claim_verdict_question` task; reframes the claim as "Is the following scientific claim supported or refuted by the literature: …?". Supports `oracle_context` and `use_retrieval` modes. |
| `experiments/configs/open_scholar_config.yaml` | YAML config for the OpenScholar baseline: S2 adaptive retrieval on, CE reranker, `top_n=10`, `claude-sonnet-4-6`. |
| `experiments/configs/random_baseline_config.yaml` | YAML config for the random baseline: 4 datasets, 10 repeats, seed=100. |
| `experiments/configs/llm_only_config.yaml` | YAML config for the LLM-only parametric baseline: `anthropic/claude-sonnet-4-6`, 3 repeats. |

## Modified Files

| File | Changes |
|---|---|
| `experiments/run_baselines_datasets.py` | Added `--os-oracle-context` flag; added `--config` YAML pre-parse via a throwaway `ArgumentParser`; added all OpenScholar-specific CLI args (`--os-model`, `--os-api`, `--os-top-n`, `--os-max-tokens`, `--os-retrieval`, `--os-reranker`). |
| `experiments/baselines/shared/evaluate.py` | Added weighted FPR and weighted FNR (weighted by per-class support) alongside the existing macro variants. Harness now detects whether `baseline.verify()` accepts a `context` kwarg via `inspect.signature` and forwards the full claim dict when it does. |
| `OpenScholar/src/open_scholar.py` | Added `print` blocks at the start of each LLM-call stage (`generate_response`, `get_feedback`, `retrieve_keywords`, `edit_with_feedback`, `edit_with_feedback_retrieval`) that emit the full prompt under a labelled `===` banner, enabling end-to-end prompt inspection without a debugger. |
| `OpenScholar/src/use_search_apis.py` | Increased `time.sleep` after `search_paper_via_query` from 0.5 s to 1.1 s to respect the S2 API 1 req/s rate limit and avoid 429 errors on consecutive keyword queries within the feedback loop. |

## Architecture

```
run_baselines_datasets.py
  └─ build_baseline("open_scholar", args)
       └─ OpenScholarBaseline(
            model="claude-sonnet-4-6", api="anthropic",
            top_n=10, max_tokens=None,   # None → OpenScholar default (3000)
            use_retrieval=True,          # --ss_retriever --feedback
            oracle_context=False,        # no CSV evidence injection
            reranker="OpenScholar/OpenScholar_Reranker",
            subprocess_timeout=600,      # doubled when retrieval is active
          )
               │
               ▼  per claim (subprocess → OpenScholar/run.py)
         ┌─────────────────────────────────────────────────────┐
         │ input: framed as question                           │
         │   "Is the following scientific claim supported      │
         │    or refuted by the literature: <claim>?           │
         │    Answer uncertain otherwise."                     │
         │                                                     │
         │ task_name: claim_verdict_question (JSON verdict)    │
         │                                                     │
         │ Call 1: generate_response (empty or oracle ctxs)   │
         │   → initial JSON verdict (zero-shot)                │
         │                                                     │
         │ Call 2: get_feedback (if use_retrieval)             │
         │   → up to 3 Feedback/Question pairs                 │
         │                                                     │
         │ Call 3: retrieve_keywords (per question)            │
         │   → S2 API queries → CE rerank                      │
         │                                                     │
         │ Call 4: edit_with_feedback_retrieval                │
         │   → revised verdict with S2 passages                │
         └─────────────────────────────────────────────────────┘
               │
               ▼
         _parse_verdict(raw_output)
           1. strip markdown fences
           2. try json.loads
           3. regex for "label": "SUPPORT|REFUTE|UNCERTAIN"  ← truncation fallback
           4. return NEI

run_baselines_datasets.py (YAML config path)
  └─ pre-parse --config path/to.yaml
       └─ yaml.safe_load → parser.set_defaults(**cfg)   ← YAML = defaults
            └─ p.parse_args()                           ← CLI flags override
```

## Key Design Decisions

- **`claim_verdict_question` task, not `claim_full`** — The baseline uses the `claim_verdict_question` task (JSON output: `label` / `reasoning` / `evidence`) rather than `claim_full` (plain `true`/`false`). This gives structured output that maps cleanly to `BaselineResult` and allows `_parse_verdict` to recover from truncated JSON via regex.

- **Claim reframed as a question** — The raw claim text is reframed as `"Is the following scientific claim supported or refuted by the literature: <claim>? Answer uncertain otherwise."` This ensures the OpenScholar feedback and keyword-extraction stages (which expect a question) produce coherent follow-up queries.

- **`oracle_context=False` as the new default** — The previous default silently injected the ground-truth CSV evidence snippet as passage `[0]` whenever a claim had an `evidence` field, making the baseline look better than it was. Separating oracle and retrieval modes with an explicit flag makes the distinction clear and preserves backward compatibility via `--os-oracle-context`.

- **Regex label recovery over hard NEI fallback** — When `max_tokens` truncates a verbose response before the JSON closing brace, the label is always emitted near the start of the object. A targeted regex (`"label"\s*:\s*"SUPPORT|REFUTE|UNCERTAIN"`) recovers the correct prediction in the majority of truncation cases without reprompting.

- **Subprocess timeout doubles when retrieval is active** — The default 300 s timeout is automatically increased to 600 s when `use_retrieval=True`, because the S2 API + feedback loop adds 2–3 extra LLM calls per claim on top of the initial generation.

- **S2 rate limit: sleep bumped to 1.1 s** — The free-tier S2 API key is capped at 1 req/s. The previous 0.5 s sleep in `search_paper_via_query` caused 429 errors on the second and third keyword queries within a feedback round. Setting it to 1.1 s gives a safe margin above the limit.

- **Prompt logging instrumentation** — Each of the five LLM call sites in `open_scholar.py` now prints the full prompt under a labelled banner (`[STAGE: X — PROMPT SENT TO LLM]`) before the API call. `open_scholar_baseline.py` passes subprocess stdout/stderr through to the calling process so these logs are visible without attaching a debugger.

- **YAML keys mirror CLI `dest` names** — The `--config` pre-parse pattern uses a throwaway `ArgumentParser(add_help=False)` to extract `--config` before the main parser runs, then applies `set_defaults(**cfg)` so any explicit CLI flag still overrides the config value. YAML keys match `argparse` `dest` names (e.g. `os_model`, `os_retrieval`) with no translation layer.

- **Weighted FPR/FNR in metrics** — Added alongside macro variants. Rare classes (NEI/UNCERTAIN) previously inflated the macro average; weighting by per-class support (tp+fn) gives a more representative picture of classification errors on the dominant classes.

- **No restart on partial run** — The evaluation harness streams results to JSONL after each claim and supports resume via `resume_path`. The initial run (8 claims with the old parser) was discarded and restarted after the regex fix was applied, since the NEI-from-truncation errors disproportionately affected SUPPORT claims.

## Bug Fixes

### 1. Truncated JSON verdict recovery (regex fallback for `_parse_verdict`)
**Symptom:** When OpenScholar's verbose reasoning exceeds the token budget, the
JSON response is cut off before the closing `}`.  `json.loads` raises
`JSONDecodeError`; the previous hard fallback returned `"NEI"`, eliminating the
actual prediction.

**Fix (implemented):** `_parse_verdict` strips markdown fences, attempts
`json.loads`, then falls back to a regex search for `"label": "SUPPORT|REFUTE|UNCERTAIN"`.
If a label is recovered, partial reasoning is also captured and returned.  Only
if no label is found does the method fall back to NEI.

---

### 2. Feedback loop corrupts JSON verdict format
**Symptom:** The feedback prompt (`instruction_feedback_prompt`) was designed
for long-form NLP survey essays.  Given a compact JSON object as input, it
generates essay-style feedback and an expanded prose answer.  `_parse_verdict`
cannot decode the prose → NEI fallback.

**Status:** Open.  Fix: disable `--feedback` for `claim_verdict_question` tasks,
or add a JSON-constrained edit prompt variant.

---

### 3. Domain-mismatched S2 keyword extraction
**Symptom:** `keyword_extraction_prompt` is hard-coded with NLP framing
("*related to the most recent NLP research*").  For SIGNOR biology claims the
model generates NLP-flavoured queries; retrieved papers are irrelevant.  Combined
with the `claim_verdict_question` instruction that prefers REFUTE over UNCERTAIN
when no relevant passages are found, noisy retrieval actively increases the
false-negative rate on SUPPORT claims.

**Status:** Open.

---

### 4. S2 API rate-limit 429 errors
**Symptom:** The free-tier S2 API is capped at 1 req/s.  The previous 0.5 s
`time.sleep` in `search_paper_via_query` caused 429 errors on the second and
third keyword queries within a feedback round.

**Fix (implemented):** Sleep increased to 1.1 s in `OpenScholar/src/use_search_apis.py`.

## Results Summary

| Run | Macro F1 | SUPPORT recall | REFUTE recall | Cost (USD) |
|---|---|---|---|---|
| oracle (CSV evidence injected) | 0.437 | 0.396 | 0.750 | 1.638 |
| test_no_adaptive (also oracle — old default) | 0.417 | 0.358 | 0.769 | 0.195 |
| **adaptive S2, no oracle** | **0.232** | **0.038** | **0.692** | **1.239** |
