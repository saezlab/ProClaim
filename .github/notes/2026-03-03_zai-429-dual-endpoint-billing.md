# Z.AI 429 Error — Dual-Endpoint Billing Investigation — 2026-03-03

**Branch:** `RLM`

## Summary

Investigated the `429 Insufficient balance` errors that caused all fact-extraction LLM calls to fail during the notebook demo (`evidence_report.ipynb`). The previous fix note (2025-02-25) attributed this to a depleted Z.AI account. The actual root cause is that Z.AI maintains **separate billing/resource packages** for its OpenAI-compatible endpoint (`/api/paas/v4/`) and its Anthropic-compatible endpoint (`/api/anthropic`). The OpenAI-compatible package is depleted; the Anthropic-compatible one is not. This means the outer Claude Agent SDK loop (which uses `/api/anthropic`) works fine, but the inner `llm()` callable used for fact extraction (which uses `/api/paas/v4/`) fails on every call.

## Key Findings

### Architecture of the failure (Mode A)

The verification system has two nested LLM call paths that hit different Z.AI endpoints:

```
┌───────────────────────────────────────────────────────────┐
│  Claude Agent SDK (outer loop)                            │
│  Endpoint: https://api.z.ai/api/anthropic                 │
│  Client:   Anthropic protocol                             │
│  Auth:     ANTHROPIC_AUTH_TOKEN = GLM_API_KEY              │
│  Status:   ✓ WORKS (ran 40+ turns successfully)           │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  llm() callable (inner — fact extraction subagents) │  │
│  │  Endpoint: https://api.z.ai/api/paas/v4/            │  │
│  │  Client:   OpenAI protocol                          │  │
│  │  Auth:     GLM_API_KEY (same key)                   │  │
│  │  Status:   ✗ 429 — "Insufficient balance"           │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### Evidence from notebook demo

- **Cell 8**: All 12 papers → `extract_and_add_facts` → `llm()` returns 429 error code 1113 on every retry → 0 facts extracted
- **Cell 13**: Agent checked env vars — `GLM_API_KEY: set`, `OPENAI_API_KEY: NOT SET`, `ANTHROPIC_API_KEY: NOT SET`
- **Cell 18**: Agent retried with explicit `base_url="https://api.z.ai/api/paas/v4/"` and `GLM_API_KEY` → same 429
- **Cell 20**: Agent tested Anthropic client against `/api/anthropic` with `ANTHROPIC_AUTH_TOKEN` (same `GLM_API_KEY` value) → **success** ("Hello! How can I help you today?")
- **Cell 21**: Agent switched the inner `llm()` to use Anthropic SDK against `/api/anthropic` → fact extraction started working

### Environment config

Only one API key exists in `.env`:
```
GLM_API_KEY=07694b...
```

The shell also has `ANTHROPIC_AUTH_TOKEN` set to the same `GLM_API_KEY` value (set by `_build_env()` in the demo script). No separate Claude/Anthropic key exists.

## Affected Files

| File | Role | Impact |
|------|------|--------|
| `scripts/verification/demo_evidence_programming.py` | Mode A entry point | Kernel prelude wires `llm()` via OpenAI client → `/api/paas/v4/` (depleted) |
| `src/pkevolve/verification/repl_orchestrator.py` | Mode B REPL loop | `verify_claim_repl()` wires the same OpenAI-based `llm()` via `KernelRunner.inject_prelude()` |
| `src/pkevolve/verification/kernel_runner.py` | Kernel prelude injection | Injects the OpenAI-based `llm()` function into the Jupyter kernel |

## Key Design Decisions

- **The 429 is NOT a code bug** — the endpoint URL (`/api/paas/v4/`) is correct (confirmed by the 2025-02-25 fix). The issue is Z.AI billing state.
- **Z.AI has separate resource packages per endpoint type** — the Anthropic-compatible endpoint (`/api/anthropic`) and OpenAI-compatible endpoint (`/api/paas/v4/`) are billed independently, despite using the same API key.
- **Previous diagnosis was incomplete** — the 2025-02-25 note attributed this to "Z.AI account balance depleted" and recommended "account needs recharging." In reality, only the OpenAI-compatible package is depleted; the Anthropic-compatible one has balance.

## Recommended Fix

Switch the inner `llm()` callable to use the Anthropic client against `/api/anthropic` instead of the OpenAI client against `/api/paas/v4/`. This requires changing:

1. **Kernel prelude** (in both `demo_evidence_programming.py` and `kernel_runner.py`): Replace the `OpenAI`-based `llm()` function with an `Anthropic`-based one targeting `/api/anthropic`
2. **`repl_orchestrator.py`**: Update `DEFAULT_BASE_URL` and the `inject_prelude()` call to use the Anthropic endpoint
3. **Alternatively**: Recharge the OpenAI-compatible resource package on Z.AI, or add a fallback chain that tries `/api/paas/v4/` first and falls back to `/api/anthropic` on 429

The cleanest long-term fix is to support both endpoint types and let the user configure which one to use, since Z.AI billing may change.
