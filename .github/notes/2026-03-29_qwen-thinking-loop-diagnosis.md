# Qwen Thinking Mode — Reasoning Loop Pathology (2026-03-29)

**Branch:** `exp/init`

## Summary

Qwen3-series models (tested: `qwen3.5-9b`, `qwen3-8b`) exhibit a catastrophic reasoning-loop failure when used as fact-extraction subagents via vLLM. On papers irrelevant to the query claim, both models enter an infinite `<think>` deliberation — consuming ~29K tokens of reasoning on whether to emit `[]` — and produce **zero output tokens**. This is a confirmed, documented pathology at the intersection of Qwen3's thinking mode and structured JSON-output tasks. The same prompt on Claude Haiku 4.5 returns `[]` with 14 output characters and zero thinking overhead.

This note consolidates evidence from the `signor_eval_test` debug logs, external issue trackers, and extended analysis at https://claude.ai/share/84e63bb1-2195-4735-85a2-588a830fe2ea.

---

## Observed Symptoms

| Model | Paper (PMID) | Thinking tokens | Output tokens | Output |
|-------|-------------|-----------------|---------------|--------|
| `qwen3.5-9b` | 23316053 | ~28,968 chars | 0 | `""` |
| `qwen3-8b` | 23316053 | ~29,096 chars | 0 | `""` |
| `claude-haiku-4-5-20251001` | 23316053 | 0 | 14 | `[]` |

PMID 23316053 is an Abl kinase minireview that mentions `c-Fes` only to distinguish it from `c-Abl`. The claim is `"FES directly inhibits BCR"`. The correct answer is `[]` — the paper has no relevant facts.

The thinking block content (captured in `workspace/debug_logs/`) shows the model cycling through approximately 30 re-evaluations of the same decision:

> *"Wait, I'll check… Actually, I should reconsider… Wait, I need to double check…"*

The model oscillates over: (1) whether `"Abl is distinct from c-Fes"` constitutes a NEUTRAL fact relevant to the claim, (2) whether `relevant_subclaims: []` is valid JSON, and (3) whether an empty array output violates `"extract every atomic fact"`. It never resolves any of these and exhausts its token budget.

---

## Root Cause Analysis

### Why Qwen3 loops on this specific task

Qwen3's 4-stage post-training pipeline includes a **thinking-mode fusion stage** that trains the model to deeply deliberate before structured output. This is well-suited for multi-step reasoning tasks. On structured extraction tasks with a **trivially empty correct answer**, the training creates a failure mode:

1. **Ambiguous-but-trivially-empty answer**: The model is told to "extract every atomic fact relevant to the claim." The paper has none. But the model's training instils doubt — maybe it's missing something?

2. **Constrained output format conflicts with thinking drive**: `"Output ONLY a JSON array of fact objects. No other text."` with no explicit empty-paper escape clause creates an underdetermined state. The model knows what format is required but cannot commit to `[]` without violating its sense of thoroughness.

3. **No thinking-budget cap**: Without `thinking_budget` set in `extra_body`, vLLM allows the model to use the entire `max_tokens` allocation (default 8,000) for `<think>` tokens. When the model loops, it exhausts the budget and emits nothing, because no `</think>` token is generated before the context limit.

4. **`_THINK_RE` in `subagents.py` cannot save it**: The regex `re.compile(r"<think>.*?</think>\s*", re.DOTALL)` at L45 strips completed thinking blocks — but when thinking never terminates (no `</think>` emitted), the entire response is an unclosed thinking block and stripping removes everything, leaving `""`.

### vLLM server configuration

`scripts/vllm_node_setup.sh` launches vLLM with `--reasoning-parser qwen3`, enabling server-side thinking mode. Neither `make_subagent_llm()` in `config.py` nor the `setup_kernel()` `make_llm()` call in `evidence_api.py` pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to disable it.

---

## Confirmed External Reports

This is a known, publicly documented issue across multiple trackers:

| Issue | Date | Description |
|-------|------|-------------|
| **QwenLM/Qwen3 #1700** | Oct 2025 | Qwen3 Instruct-2507 models don't terminate in vLLM when generating structured output — even for `"What is 4+4?"`. Produces `LengthFinishReasonError` with `completion_tokens=4081`, `output_tokens=0`. |
| **QwenLM/Qwen3 #1384** | May 2025 | Qwen3-30B-A3B loops/repeats indefinitely until `max_tokens`. Qwen team officially recommends `presence_penalty=1.5`. |
| **QwenLM/Qwen3 #1817** | ~2025 | Thinking mode plans tool calls but fails to emit them ~60% of the time (Qwen3-32B-AWQ + vLLM). Thinking block "satisfies" the model's intent without producing output. |
| **vLLM #23074** | ~2025 | `thinking=True` + `guided_json` → output content is always empty. Confirms exact symptom. |
| **vLLM #18819** | May 2025 | Structured output (guided decoding) with Qwen3 is broken when `enable_thinking=False` via `chat_template_kwargs`, but `/no_think` system-prompt suffix produces valid JSON (with a spurious extra newline). |
| **vLLM #35574** | Feb 2026 | `enable_thinking=False` via `chat_template_kwargs` does **not** disable thinking for Qwen3.5 models — the flag is silently ignored. |
| **QwenLM/Qwen3 discussion #1744** | ~2025 | Qwen team confirms `presence_penalty=1.5` as official mitigation for repetition/looping during thinking. The model card also explicitly lists this recommendation. |

---

## Fix Priority Matrix

| Fix | Mechanism | Reliability | Notes |
|-----|-----------|-------------|-------|
| Append `/no_think` to system prompt | Soft instruction in system message | **High** — confirmed working in vLLM #18819 | Works across Qwen3 and Qwen3.5; minor trailing newline issue |
| `presence_penalty=1.5` | Sampling parameter | **Medium** — reduces loops, does not eliminate | Officially recommended by Qwen team; too high causes incoherence |
| `extra_body: {chat_template_kwargs: {enable_thinking: false}}` | Chat template kwarg | **Unreliable** — broken for Qwen3.5 per vLLM #35574 | Works on Qwen3 base but not Qwen3.5 |
| `thinking_budget: N` in `extra_body` | Token cap on thinking block | **Medium** — prevents exhaustion, model still loops | Prevents context starvation but doesn't stop loop |
| Add explicit empty-answer instruction to prompt | Prompt engineering | **High for this task** | Resolves ambiguity at source; model-agnostic |

### Recommended combination for `extract_facts()`

1. **Add explicit empty-array instruction** to the `extract_facts` prompt in `subagents.py`:
   > *"If the paper contains no facts relevant to the claim or its subclaims, output an empty JSON array `[]` immediately without deliberation."*

2. **Append `/no_think` to the system message** in `make_llm()` calls used for subagents — or construct the system prompt in `subagents.py` with `/no_think` suffix.

3. **Add `presence_penalty=1.5`** to the `make_subagent_llm()` factory in `config.py` as a belt-and-suspenders measure.

These three together are model-agnostic, don't depend on the unreliable `enable_thinking=False` flag, and address both the path that causes the loop (underdetermined prompt) and the path that perpetuates it (no sampling pressure to terminate).

---

## Planned Code Changes

### `src/pkevolve/verification/config.py` — `make_subagent_llm()`

```python
# Current
def make_subagent_llm(self):
    from pkevolve.verification.llm_factory import make_llm
    return make_llm(
        base_url=self.subagent_base_url,
        api_key=self.api_key,
        model=self.subagent_model,
    )

# Proposed
def make_subagent_llm(self):
    from pkevolve.verification.llm_factory import make_llm
    return make_llm(
        base_url=self.subagent_base_url,
        api_key=self.api_key,
        model=self.subagent_model,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        presence_penalty=1.5,
    )
```

Note: `chat_template_kwargs: {enable_thinking: false}` is unreliable on Qwen3.5 (vLLM #35574), so `/no_think` in the prompt is the primary disable mechanism. `presence_penalty` prevents looping if thinking is still active.

### `src/pkevolve/verification/subagents.py` — `extract_facts()` prompt

Add to the Rules section:
```
- If the paper contains no facts relevant to the claim or any of its subclaims,
  output an empty JSON array [] immediately. Do not deliberate over this decision.
```

### `src/pkevolve/verification/evidence_api.py` — `setup_kernel()` (`make_llm()` call)

Mirror the `presence_penalty=1.5` and `extra_body` changes from `config.py`, since `setup_kernel()` constructs its own `make_llm()` independently from environment variables.

---

## Safety Considerations

- **Non-Qwen models**: `extra_body` with `chat_template_kwargs` is silently ignored by GLM, Llama, and Mistral backends — no adverse effect.
- **`presence_penalty=1.5` on non-looping models**: Slightly reduces verbosity; not harmful for extraction tasks that should output compact JSON.
- **Prompt-level fix is universally safe**: Adding the explicit empty-array instruction improves behavior on all models, not just Qwen.
- **Reversibility**: The `enable_thinking` flag can be re-enabled per-call via `extra_body` override if a specific task (e.g., complex multi-hop synthesis) benefits from extended reasoning.

---

## Related Notes

- [2026-03-25 SIGNOR eval test sample run issues](.github/notes/2026-03-25_signor-eval-test-sample-run-issues.md) — full 6-issue analysis of the test run; Issue 1 in that note is the same pathology documented here with additional context on paper attrition and subagent quality gaps.
