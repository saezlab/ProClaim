# Implementation Plan: Standardize Token Tracking in `evidence_programming_direct.py`

## Goal

Replace the hand-rolled `total_usage` dict in `evidence_programming_direct.py`
with the same `CostTracker`-based approach used by all baselines.
Cache-token accounting is deliberately dropped — the point is apples-to-apples
comparison with baselines, which also don't account for cache pricing.

---

## Current state

`evidence_programming_direct.py` (lines 704-794) manually accumulates five
LiteLLM keys and never computes cost:

```python
total_usage = {
    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    "cache_read_tokens": 0, "cache_creation_tokens": 0,
}
# ... per call:
total_usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
total_usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
...
```

Baselines use `CostTracker.record(action, in_tok, out_tok)` and read
`tracker.summary()` which includes `cost_usd` and `num_llm_calls`.

---

## Plan

### Step 1 — Move `CostTracker` to `src/pkevolve/`

Create `src/pkevolve/verification/cost_tracker.py` by copying
`experiments/baselines/shared/cost_tracker.py` verbatim (no changes to the
class itself).

Replace the baselines file content with a one-line re-export so all existing
imports keep working without touching any baseline file:

```python
# experiments/baselines/shared/cost_tracker.py
from pkevolve.verification.cost_tracker import CostTracker, TraceEntry  # noqa: F401
```

---

### Step 2 — Refactor `verify_claim_direct()`

**Replace the dict initialisation (lines 703-710):**

```python
# remove:
total_usage = {"prompt_tokens": 0, "completion_tokens": 0, ...}

# add:
from pkevolve.verification.cost_tracker import CostTracker
tracker = CostTracker(model=agent_model)
```

**Replace the per-call accumulation block (lines 788-794):**

```python
# remove:
total_usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
...

# add:
if hasattr(response, "usage") and response.usage:
    u = response.usage
    tracker.record(
        "llm_call",
        input_tokens=getattr(u, "prompt_tokens", 0) or 0,
        output_tokens=getattr(u, "completion_tokens", 0) or 0,
    )
```

Cache tokens are read from LiteLLM but intentionally not passed to
`record()`.  They won't appear in cost — matching baseline behaviour.

**Replace the save block (lines 854-859):**

```python
# remove:
logger.info("Total token usage: %s", json.dumps(total_usage))
usage_path = output_dir / "token_usage.json"
usage_path.write_text(json.dumps(total_usage, indent=2))

# add:
summary = tracker.summary()
logger.info("Total token usage: %s", json.dumps(summary))
usage_path = output_dir / "token_usage.json"
usage_path.write_text(json.dumps(summary, indent=2))
```

---

### Step 3 — Update downstream readers (key rename)

`CostTracker.summary()` outputs `input_tokens` / `output_tokens`.
The current `token_usage.json` uses `prompt_tokens` / `completion_tokens`.
Two reader functions need a backward-compat fallback so old result dirs still parse:

**`experiments/run_signor_eval.py` and `experiments/run_connectomedb_eval.py`
— `_parse_tokens_from_usage_json()` in each:**

```python
# before:
"input_tokens": data.get("prompt_tokens", 0),
"output_tokens": data.get("completion_tokens", 0),

# after:
"input_tokens": data.get("input_tokens", data.get("prompt_tokens", 0)),
"output_tokens": data.get("output_tokens", data.get("completion_tokens", 0)),
```

---

## New `token_usage.json` schema

```json
{
  "input_tokens": 1234,
  "output_tokens": 456,
  "cost_usd": 0.010614,
  "latency_seconds": 42.1,
  "num_llm_calls": 8
}
```

`total_tokens`, `cache_read_tokens`, `cache_creation_tokens` are dropped
from the file.

---

## Files touched

| File | Change |
|---|---|
| `src/pkevolve/verification/cost_tracker.py` | **New** — canonical location (copy of baselines version) |
| `experiments/baselines/shared/cost_tracker.py` | Replace with 1-line re-export shim |
| `src/pkevolve/verification/evidence_programming_direct.py` | Use CostTracker, drop manual dict |
| `experiments/run_signor_eval.py` | Backward-compat key fallback in reader |
| `experiments/run_connectomedb_eval.py` | Same |

`scripts/eval_signor_direct_csv.py` — imports from the re-export shim,
reads `cache_*` keys from result CSVs (not `token_usage.json`), no change needed.
