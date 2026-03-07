# Claude Code Project Instructions

This file is read automatically by Claude Code at session start.
See also: `.github/copilot-instructions.md` for general project conventions.
See also: `doc/sdk_vs_repl_modes.md` for SDK vs REPL mode comparison,
transparency analysis, and academic reproducibility guidance.

## Claude Agent SDK -- GLM API Configuration

Scripts that invoke Claude via the **Claude Agent SDK** (`claude_agent_sdk`) connect
to the **GLM backend** at `api.z.ai`, which exposes a native Anthropic-compatible
`/v1/messages` endpoint.

### Architecture (default — direct connection)

```
Claude CLI binary
  |  (Anthropic Messages API)
  v
GLM Backend (api.z.ai/api/anthropic)  ← native Anthropic Messages API
```

The `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` env var suppresses most
proprietary headers that would otherwise be rejected. This is set as
defense-in-depth in all scripts.

### Architecture (optional — LiteLLM proxy)

For cases where the GLM endpoint rejects beta headers, a local LiteLLM proxy
can be used as a translation gateway. Enable with `--use-litellm` or `USE_LITELLM=1`.

```
Claude CLI binary
  |  (Anthropic Messages API + proprietary headers)
  v
LiteLLM Proxy (localhost:4000)
  |  (clean Anthropic Messages API)
  v
GLM Backend (api.z.ai/api/anthropic)
```

### Required environment variables

| Variable | Purpose | Default |
|---|---|---|
| `GLM_API_KEY` | API key for `api.z.ai` | *(in `.env`)* |
| `API_TIMEOUT_MS` | Request timeout in milliseconds | `3000000` (50 min) |

Only needed when using LiteLLM proxy (`--use-litellm`):

| Variable | Purpose | Default |
|---|---|---|
| `LITELLM_MASTER_KEY` | Auth key for CLI-to-proxy traffic | `sk-litellm-local-dev` |
| `LITELLM_PORT` | LiteLLM proxy port | `4000` |

All variables live in `.env` at project root (gitignored).

### Script env pattern

All Claude Agent SDK scripts use the direct GLM connection by default:

```python
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

GLM_API_BASE = "https://api.z.ai/api/anthropic"
GLM_API_KEY = os.getenv("GLM_API_KEY")

env = {
    **os.environ,
    "ANTHROPIC_AUTH_TOKEN": GLM_API_KEY,
    "ANTHROPIC_BASE_URL": GLM_API_BASE,
    "ANTHROPIC_API_KEY": "",  # Must be empty to prevent Anthropic auth
    "API_TIMEOUT_MS": os.getenv("API_TIMEOUT_MS", "3000000"),
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
}

options = ClaudeAgentOptions(
    model="glm-4.6",
    env=env,
    cwd=str(PROJECT_ROOT),
    ...
)
```

### Key rules

1. **Never hardcode credentials.** `GLM_API_KEY` lives in `.env`.
2. **Use `ANTHROPIC_AUTH_TOKEN`** (not `ANTHROPIC_API_KEY`) for direct GLM connection.
3. **Set `ANTHROPIC_API_KEY` to empty string** to prevent the CLI from trying Anthropic auth.
4. **Always set `API_TIMEOUT_MS` high** (at least `3000000` / 50 min) for agent loops.
5. The beta-suppression env vars are set as **defense-in-depth**.
6. The `.env` file is gitignored. Copy `.env.example` and fill in credentials.

### Starting the LiteLLM proxy (optional)

Only needed when using `--use-litellm`:

```bash
./scripts/start_litellm.sh          # foreground
./scripts/start_litellm.sh --bg     # background (logs to litellm.log)
```

### Model names

When connecting directly to GLM, use GLM model names:

| Model name | Anthropic endpoint | OpenAI-compatible endpoint |
|---|---|---|
| `glm-5` | `api.z.ai/api/anthropic` | `api.z.ai/api/paas/v4/` |
| `glm-4.7` | `api.z.ai/api/anthropic` | `api.z.ai/api/paas/v4/` |
| `glm-4.6` | `api.z.ai/api/anthropic` | `api.z.ai/api/paas/v4/` |
| `glm-4.5-air` | `api.z.ai/api/anthropic` | `api.z.ai/api/paas/v4/` |

**Important**: The OpenAI-compatible endpoint is `https://api.z.ai/api/paas/v4/`
(the `openai` Python SDK appends `/chat/completions`). Do NOT use
`/api/openai` — that path does not work.

When using LiteLLM proxy, mapped CLI names also work:

| CLI model name | Maps to | Endpoint |
|---|---|---|
| `claude-sonnet-4-5-20250929` | `glm-5` | `api.z.ai/api/anthropic` |
| `glm-4.6` | `glm-4.6` | `api.z.ai/api/anthropic` |

### Scripts using this pattern

- `src/pkevolve/verification/evidence_programming.py` -- evidence verification with notebook output
- `scripts/claude_sdk/run_signor_qa_glm.py` -- batch SIGNOR QA via GLM models
- `scripts/claude_sdk/run_signor_qa_claude.py` -- batch SIGNOR QA via Claude
- `src/pkevolve/verification/orchestrator.py` -- evidence programming orchestrator
