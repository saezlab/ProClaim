# OpenScholar Baseline — SIGNOR Run Results — 2026-04-04

**Branch:** `exp`

## Summary

Integrated the OpenScholar RAG pipeline as a new baseline (`open_scholar`) in the evaluation framework. The baseline wraps OpenScholar as a subprocess, framing each claim as a literature question and routing it through the `claim_verdict` task. Pre-existing evidence from the SIGNOR CSV is forwarded as a retrieved passage via `--use_contexts`. The full SIGNOR dataset (111 claims) was evaluated using Claude Sonnet 4.6, producing results in the same `BaselineResult` JSONL schema as the existing `llm_only` and `random` baselines.

## New Files

| File | Purpose |
|------|---------|
| `experiments/baselines/open_scholar_baseline.py` | `OpenScholarBaseline` class — subprocess wrapper around OpenScholar's `run.py`, routing claims through `claim_verdict` task with configurable model/API/retrieval flags |
| `results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.jsonl` | Per-claim predictions (111 rows, `BaselineResult` JSON lines) |
| `results/baselines/open_scholar/claude-sonnet-4-6/signor_metrics.json` | Aggregated metrics (accuracy, macro F1, binary F1, per-class stats, cost) |

## Modified Files

| File | Changes |
|------|---------|
| `experiments/run_baselines_datasets.py` | Registered `open_scholar` in `build_baseline()` factory; added 5 `--os-*` CLI flags (`--os-model`, `--os-api`, `--os-top-n`, `--os-max-tokens`, `--os-retrieval`); `load_claims()` now preserves extra CSV columns (`evidence`, `pmid`, `entity_a`, `entity_b`, `effect`) for richer baselines; `n_repeats` and `baseline_subdir` logic updated for `open_scholar` |
| `experiments/baselines/shared/evaluate.py` | `EvaluationHarness.run()` uses `inspect.signature()` to detect if a baseline's `verify()` accepts a `context` kwarg, and forwards the full claim dict automatically — backward-compatible with all existing baselines |
| `experiments/README.md` | Added OpenScholar to baselines table, quick-start examples, CLI flags reference, prerequisites section, `BaselineResult` field reference, updated "Adding a new baseline" guide; removed references to deleted `run_baselines_signor.py` |

## Architecture

```
run_baselines_datasets.py
    │
    ├─ load_claims(csv)         CSV → list[claim_dict] (with evidence/pmid columns)
    │
    ├─ build_baseline("open_scholar", args)
    │       └─ OpenScholarBaseline
    │               model=claude-sonnet-4-6, api=anthropic
    │
    └─ EvaluationHarness.run(claims)
            │
            ├─ verify(claim_id, claim, gold, context={evidence, pmid, ...})
            │       │
            │       ├─ Frame claim as literature question
            │       ├─ Inject CSV evidence as ctxs[0] if available
            │       ├─ Write temp input.jsonl
            │       ├─ subprocess: OpenScholar/run.py
            │       │       --task_name claim_verdict
            │       │       --use_contexts (when evidence present)
            │       │       --zero_shot --top_n 5
            │       ├─ Parse JSON verdict from output
            │       └─ Return BaselineResult
            │
            └─ metrics() + save() → JSONL + metrics JSON
```

## Retrieval & Prompt Configuration

### CLI parameters passed to OpenScholar `run.py`

```
python run.py \
  --input_file <tmpdir>/input.jsonl \
  --output_file <tmpdir>/output.json \
  --api anthropic \
  --model_name claude-sonnet-4-6 \
  --task_name claim_verdict \
  --zero_shot \
  --use_contexts \          # enabled because CSV evidence was injected
  --top_n 5 \
  --max_tokens 1500
```

**Flags NOT used:** `--ss_retriever`, `--feedback`, `--ranking_ce`, `--posthoc_at`, `--reranker`

### How evidence reaches the LLM

1. **`open_scholar_baseline.py`** reads the `evidence` and `pmid` columns from the SIGNOR CSV for each claim.
2. It constructs a single context entry: `{"title": "PMID:17652154", "text": "Because adenylyl cyclases are directly activated by..."}` and writes it to `input.jsonl` as `ctxs[0]`.
3. The `--use_contexts` flag tells `OpenScholar.run()` to treat `ctxs` as pre-retrieved passages (skipping any live retrieval).
4. Inside `generate_response()`, since `task_name="claim_verdict"` is in `instructions.task_instructions`, and `zero_shot=True`, the prompt is assembled as:

```
{task_instruction}          ← claim_verdict system prompt (verdict definitions + rules)
References:
[0] Title: PMID:17652154 Text: <evidence snippet from SIGNOR CSV>
                            ← formatted by: "[{idx}] Title: {title} Text: {text}\n"
Claim: <framed question>   ← "What does the scientific evidence say about <claim>..."
```

5. This single prompt is sent to Claude Sonnet 4.6 via LiteLLM (`anthropic/claude-sonnet-4-6`) with `temperature=0.7`, `max_tokens=1500`.
6. The raw output is the JSON verdict (`{"label": ..., "reasoning": ..., "evidence": [...]}`), which `[Response_Start]`/`[Response_End]` markers are stripped from if present.

### What was NOT used

| Feature | Flag | Why skipped |
|---------|------|-------------|
| **S2 keyword retrieval** | `--ss_retriever` | Requires S2 API key; anonymous access returned 429 Too Many Requests |
| **Feedback loop** | `--feedback` | Only activates with `--ss_retriever`; triggers LLM self-critique → keyword extraction → S2 search → rerank → answer editing (1–3 extra LLM calls) |
| **Cross-encoder reranking** | `--ranking_ce` | Not needed — only 1 passage per claim, nothing to rerank |
| **Post-hoc attribution** | `--posthoc_at` | Adds per-sentence citation insertion; unnecessary for structured JSON verdict |
| **Passage deduplication / max_per_paper** | `--max_per_paper` | Only 1 passage, no duplicates |

### Per-claim evidence characteristics

Each SIGNOR claim has exactly **1 evidence passage** from the CSV — a sentence-length excerpt from the PMID paper that originally supported the SIGNOR edge annotation. This is a **gold evidence** setting (the passage is the curator's basis for the annotation), which differs from a true open-retrieval RAG scenario. The key limitation: when the evidence sentence doesn't directly mention the claim's entity pair or effect direction, the model defaults to REFUTE per the prompt's instruction "if no passage mentions the entities or relationship in the claim, prefer REFUTE over UNCERTAIN."

---

## SIGNOR Run Results (111 claims, Claude Sonnet 4.6)

### Headline Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 54.1% (60/111) |
| Macro F1 | 0.417 |
| Binary F1 (SUPPORT vs REFUTE) | 0.771 |
| Binary Precision | 0.983 |
| Binary Recall | 0.634 |
| Total cost | $0.195 |
| Avg latency | 9.8s/claim |

### Per-Class Metrics

Note: both the SIGNOR dataset and the OpenScholar `claim_verdict` prompt use `UNCERTAIN` as the third label. The evaluation harness internally maps `UNCERTAIN → NEI` via `normalize_label()`, but results below use the original label for clarity.

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| SUPPORT | 0.864 | 0.358 | 0.507 |
| REFUTE | 0.597 | 0.769 | 0.672 |
| UNCERTAIN | 0.045 | 0.167 | 0.071 |

### Confusion Matrix

| Gold ↓ \ Pred → | SUPPORT | REFUTE | UNCERTAIN |
|-----------------|---------|--------|-----|
| SUPPORT (53) | **19** | 24 | 10 |
| REFUTE (52) | 1 | **40** | 11 |
| UNCERTAIN (6) | 2 | 3 | **1** |

### Label Distribution

| | SUPPORT | REFUTE | UNCERTAIN |
|--|---------|--------|-----|
| Gold | 53 | 52 | 6 |
| Predicted | 22 | 67 | 22 |

## Key Design Decisions

- **Subprocess isolation**: OpenScholar runs in its own `.venv` (Python 3.12, uv-managed) via `subprocess.run()`, avoiding dependency conflicts with the grn-llm-correct environment. Each claim spawns one subprocess (~10s wall-clock including process overhead).
- **Pre-existing evidence forwarding**: Rather than relying on live S2 API retrieval (which returned 429s under anonymous access), the SIGNOR CSV's `evidence` column is injected as a single retrieved passage. This gives OpenScholar grounded context comparable to the gold evidence setting.
- **No adaptive S2 retrieval in this run**: The `--ss_retriever --feedback` flags (which trigger OpenScholar's iterative keyword extraction → S2 search → rerank → edit loop) were NOT used. The model only saw the `claim_verdict` system prompt + the single SIGNOR evidence passage. Enable with `--os-retrieval` when an S2 API key is available.
- **Backward-compatible `context` forwarding**: The `EvaluationHarness` uses `inspect.signature()` to detect whether `verify()` accepts `context`, so existing baselines (`random`, `llm_only`) are unaffected.
- **Same output schema**: Results are `BaselineResult` JSONL, identical to `llm_only`, enabling direct comparison. `input_tokens`/`output_tokens` are 0 since OpenScholar tracks cost in USD rather than tokens.

## Observations

- **Strong REFUTE bias**: The model predicted REFUTE 67 times vs 52 gold — the `claim_verdict` prompt instructs "if no passage mentions the entities or relationship, prefer REFUTE over UNCERTAIN", which drives misclassification of SUPPORT claims as REFUTE (24/53 = 45%).
- **Low SUPPORT recall (35.8%)**: 34 of 53 SUPPORT claims were missed. When the single evidence passage is ambiguous or brief, the model defaults to REFUTE.
- **High SUPPORT precision (86.4%)**: When the model does predict SUPPORT, it's almost always correct (19/22).
- **UNCERTAIN is near-random**: Only 6 gold UNCERTAIN claims and the model has no systematic way to distinguish them — 1/6 correct.
- **Binary F1 is strong (0.771)**: Excluding UNCERTAIN, the SUPPORT-vs-REFUTE task performs well, driven by high REFUTE recall (76.9%).
