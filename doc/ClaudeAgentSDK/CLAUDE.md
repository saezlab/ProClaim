# Claude Code Project Instructions

This file is read automatically by Claude Code at session start.
See also: `.github/copilot-instructions.md` for general project conventions.
See also: `doc/sdk_vs_repl_modes.md` for SDK vs REPL mode comparison,
transparency analysis, and academic reproducibility guidance.

## Claude Agent SDK — Anthropic API Configuration

Scripts that invoke Claude via the **Claude Agent SDK** (`claude_agent_sdk`) connect
to an Anthropic-compatible backend that exposes a native `/v1/messages` endpoint.

### Architecture (default — direct connection)

```
Claude CLI binary
  |  (Anthropic Messages API)
  v
Anthropic-compatible backend  ← native Anthropic Messages API
```

The `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` env var suppresses most
proprietary headers that would otherwise be rejected. This is set as
defense-in-depth in all scripts.

### Architecture (optional — LiteLLM proxy)

For cases where the backend rejects beta headers, a local LiteLLM proxy
can be used as a translation gateway. Enable with `--use-litellm` or `USE_LITELLM=1`.

```
Claude CLI binary
  |  (Anthropic Messages API + proprietary headers)
  v
LiteLLM Proxy (localhost:4000)
  |  (clean Anthropic Messages API)
  v
Anthropic-compatible backend
```

### Required environment variables

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_AUTH_TOKEN` | API key for the Anthropic-compatible backend | *(in `.env`)* |
| `ANTHROPIC_BASE_URL` | Base URL of the Anthropic-compatible backend | *(in `.env`)* |
| `API_TIMEOUT_MS` | Request timeout in milliseconds | `3000000` (50 min) |

Only needed when using LiteLLM proxy (`--use-litellm`):

| Variable | Purpose | Default |
|---|---|---|
| `LITELLM_MASTER_KEY` | Auth key for CLI-to-proxy traffic | `sk-litellm-local-dev` |
| `LITELLM_PORT` | LiteLLM proxy port | `4000` |

All variables live in `.env` at project root (gitignored).

### Script env pattern

All Claude Agent SDK scripts use the direct connection by default:

```python
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN")

env = {
    **os.environ,
    "ANTHROPIC_AUTH_TOKEN": ANTHROPIC_AUTH_TOKEN,
    "ANTHROPIC_BASE_URL": ANTHROPIC_BASE_URL,
    "ANTHROPIC_API_KEY": "",  # Must be empty to prevent default Anthropic auth
    "API_TIMEOUT_MS": os.getenv("API_TIMEOUT_MS", "3000000"),
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
}

options = ClaudeAgentOptions(
    model="<model-name>",
    env=env,
    cwd=str(PROJECT_ROOT),
    ...
)
```

### Key rules

1. **Never hardcode credentials.** `ANTHROPIC_AUTH_TOKEN` lives in `.env`.
2. **Use `ANTHROPIC_AUTH_TOKEN`** (not `ANTHROPIC_API_KEY`) for direct backend connections.
3. **Set `ANTHROPIC_API_KEY` to empty string** to prevent the CLI from trying default Anthropic auth.
4. **Always set `API_TIMEOUT_MS` high** (at least `3000000` / 50 min) for agent loops.
5. The beta-suppression env vars are set as **defense-in-depth**.
6. The `.env` file is gitignored. Copy `.env.example` and fill in credentials.

### Starting the LiteLLM proxy (optional)

Only needed when using `--use-litellm`:

```bash
./scripts/start_litellm.sh          # foreground
./scripts/start_litellm.sh --bg     # background (logs to litellm.log)
```

### Scripts using this pattern

- `src/pkevolve/verification/evidence_programming.py` -- evidence verification with notebook output
- `scripts/claude_sdk/run_signor_qa_claude.py` -- batch SIGNOR QA via Claude
- `src/pkevolve/verification/orchestrator.py` -- evidence programming orchestrator
