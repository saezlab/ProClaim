# Subagent Model Replacement Decision (2026-03-29)

**Branch:** `exp/init`

## Summary

Following the Qwen3 thinking-loop diagnosis (see [2026-03-29_qwen-thinking-loop-diagnosis.md](2026-03-29_qwen-thinking-loop-diagnosis.md)), a decision was made to replace `qwen3.5-9b` / `qwen3-8b` as the subagent model rather than investing time patching the thinking-loop pathology. This note documents the candidate evaluation process, the requirements matrix, and the chosen replacement strategy.

---

## Decision: Do Not Fix Qwen

Fixing Qwen is technically possible (see the Fix Priority Matrix in the diagnosis note) but deprioritised because:

1. **Multiple interacting bugs** — The pathology involves at least three independent issues: (a) Qwen3's thinking mode fusing with structured output, (b) vLLM's `enable_thinking=False` flag being silently ignored for Qwen3.5 (vLLM #35574), and (c) no thinking-budget cap in the current `make_llm()` factory. A robust fix requires changes across prompts, sampling params, and vLLM configuration — fragile and model-specific.

2. **Evaluation timeline pressure** — The immediate priority is running the SIGNOR evaluation pipeline end-to-end to produce baseline comparison numbers. Debugging a model-specific pathology delays this.

3. **Qwen3 8–9B is not competitive** — Even if fixed, Qwen3.5-9B at 9B parameters is outperformed by larger cloud models (gpt-oss-20b, Claude Haiku 4.5) on instruction following and JSON reliability, which are the two critical subagent requirements.

---

## Subagent Requirements

The verification system has 5 subagent functions (`src/pkevolve/verification/subagents.py`), 4 of which require structured JSON output:

| Function | Output Format | Complexity | Key Requirement |
|----------|--------------|------------|-----------------|
| `extract_facts()` | JSON array of `Fact` objects | Moderate | Up to 16K chars input (paper text) |
| `synthesize_subclaim()` | Plain text (<200 words) | Low-moderate | Free-form; no JSON constraint |
| `detect_conflicts()` | JSON array of `Conflict` objects | Low | Small input; trivial JSON |
| `identify_gaps()` | JSON array of `Gap` objects (8 enum types) | **High** | Reasoning over claim tree + evidence stats |
| `formulate_gap_queries()` | JSON array of strings | Low | Robust fallback already exists |

The 7 evaluation criteria for candidate models:

| ID | Requirement | Weight |
|----|-------------|--------|
| R1 | Reliable structured JSON output (4/5 tasks) | Critical |
| R2 | No thinking-mode pathology / clean output | Critical |
| R3 | Scientific/biomedical text comprehension | High |
| R4 | Fits on 1–2 A100 80 GB GPUs (if local) | High (local only) |
| R5 | vLLM native support with tool-call parser | High (local only) |
| R6 | Good instruction following at low temperature (0.1–0.2) | High |
| R7 | Handles ≥16K-char inputs (paper full text) | High |

---

## Candidates Evaluated

### Cloud models (no GPU required)

| Model | Provider | Params | Context | MMLU-Pro | JSON Native | Notes |
|-------|----------|--------|---------|----------|-------------|-------|
| **gpt-oss-20b** | Z.AI | 20B | Large | — | No (but instruction-tuned) | Available via `https://api.z.ai/api/paas/v4/`; uses existing `GLM_API_KEY` |
| **claude-haiku-4-5** | Anthropic | — | 200K | — | Yes (tool use) | Proven on extract_facts (returns `[]` correctly); $1/$5 per 1M tokens |

### Local models (vLLM on A100)

| Model | Params | Context | MMLU-Pro | JSON Native | Fits 1×A100? | vLLM Parser | Notes |
|-------|--------|---------|----------|-------------|-------------|-------------|-------|
| **Mistral Small 3.1 24B** | 24B dense | 128K | **80.6** | ✅ native func-call + JSON mode | ✅ ~55 GB bf16 | `mistral` (already in `vllm_node_setup.sh`) | "Agent-Centric" — designed for tool use and JSON output. Apache 2.0. |
| **Gemma 3 27B-it** | 27B dense | 128K | 76.9 | ❌ relies on guided decoding | ✅ ~56 GB bf16 | `Gemma3ForCausalLM` | Strong general model; no native JSON training |
| **Llama 4 Scout 17B×16E** | 109B total / 17B active | 10M | 74.3 | ❌ | ⚠️ int4: ~60 GB; bf16: ~220 GB | `mllama4` | MoE KV-cache overhead risky on 1 GPU |
| Phi-4 14B | 14B dense | **16K** | ~72 | ❌ | ✅ ~30 GB | ✅ | **Eliminated** — 16K context too short for paper text |
| Phi-4-mini 3.8B | 3.8B | 128K | 52.8 | ⚠️ | ✅ ~8 GB | ✅ | **Eliminated** — too small for `identify_gaps` reasoning |
| Llama 4 Maverick 17B×128E | 400B total | 1M | 80.5 | ❌ | ❌ ~800 GB bf16 | ✅ | **Eliminated** — far exceeds 2×A100 capacity |

### Key benchmarks (where available)

| Model | MMLU-Pro | IFEval | GPQA Diamond | HumanEval |
|-------|----------|--------|--------------|-----------|
| Mistral Small 3.1 24B | 80.6 | — | — | — |
| Gemma 3 27B-it | 76.9 | — | — | — |
| Llama 4 Scout | 74.3 | — | 57.2 | — |
| Phi-4 14B | — | — | 56.1 | 82.6 |

---

## Chosen Strategy (Phased)

### Phase 1: Cloud subagents (immediate — for SIGNOR evaluation)

**Primary:** `gpt-oss-20b` via Z.AI OpenAI-compatible endpoint
**Fallback:** `claude-haiku-4-5` via Z.AI or Anthropic directly

Rationale:
- Zero infrastructure setup — no vLLM server needed
- Uses the existing `GLM_API_KEY` and `https://api.z.ai/api/paas/v4/` endpoint
- Both models are proven to handle the subagent tasks without pathological loops
- Claude Haiku was already validated on the exact `extract_facts` prompt that kills Qwen3 (returns `[]` in 14 chars)
- Removes the GPU scheduling bottleneck — the SIGNOR evaluation can run on any node

Config changes (to be applied):
```yaml
llm:
  subagent_model: gpt-oss-20b          # or claude-haiku-4-5
  subagent_base_url: "https://api.z.ai/api/paas/v4/"
```

### Phase 2: Local SLM (if time permits)

**Top candidate:** `mistralai/Mistral-Small-3.1-24B-Instruct-2503`

Rationale:
- Best-in-class agentic capabilities with native function calling and JSON output — directly addresses R1 (JSON reliability)
- MMLU-Pro 80.6 — strongest benchmark score in the ≤27B weight class, beating Gemma 3 (76.9) and Llama 4 Scout (74.3)
- Fits on 1×A100 80 GB in bf16 (~55 GB), leaving headroom for 128K context KV cache
- `vllm_node_setup.sh` already has a `mistral*` parser case — no script changes needed
- Apache 2.0 license — no research restrictions
- vLLM recommended launch: `vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice`

**Runner-up:** Gemma 3 27B-it — solid alternative if Mistral underperforms on biomedical prompts; would rely on vLLM guided decoding (`response_format={"type": "json_schema", ...}`) for JSON enforcement.

### vLLM structured output as safety net

Regardless of model choice, vLLM's guided decoding can enforce JSON schema at the decoding level:
```python
response_format={"type": "json_schema", "json_schema": {"name": "facts", "schema": {...}}}
```
Backends: `xgrammar` (default), `guidance`, `outlines`. This provides model-agnostic JSON reliability for any Phase 2 local model.

---

## Cost Implications

| Model | Input ($/1M tok) | Output ($/1M tok) | Est. cost per edge† |
|-------|-------------------|--------------------|--------------------|
| gpt-oss-20b | ~0.50 | ~2.00 | ~$0.01–0.03 |
| claude-haiku-4-5 | 1.00 | 5.00 | ~$0.02–0.05 |
| Mistral Small 3.1 (local) | 0 (GPU time) | 0 (GPU time) | GPU hours only |

†Rough estimate based on ~3–5 subagent calls per paper, ~3 papers per edge.

For the SIGNOR evaluation (~200 edges), cloud subagent cost is estimated at $2–10 total — negligible compared to the outer Claude Sonnet agent cost.

---

## What This Does NOT Change

- **Outer agent model** remains `claude-sonnet-4-6` via Anthropic API — unchanged
- **Prompt templates** in `subagents.py` — unchanged (the prompts are model-agnostic)
- **`_clean_llm_json()` and `_extract_json_array()`** — retained as defensive parsing; these still protect against markdown fences and minor formatting issues from any model
- **`make_llm()` factory** — unchanged; `extra_body` parameter already exists for future model-specific tuning
- **Qwen fixes** — deferred indefinitely; the diagnosis note documents the fix path if ever needed

---

## Related Notes

- [2026-03-29 Qwen Thinking Loop Diagnosis](2026-03-29_qwen-thinking-loop-diagnosis.md) — root cause analysis and fix options
- [2026-03-25 SIGNOR Eval Test Sample Run Issues](2026-03-25_signor-eval-test-sample-run-issues.md) — original test run where the pathology was discovered
- [2026-03-28 Baseline Evaluation Pipeline](2026-03-28_baseline-evaluation-pipeline.md) — the evaluation pipeline that drives the timeline pressure
