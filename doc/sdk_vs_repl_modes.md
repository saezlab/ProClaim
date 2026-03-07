# SDK vs REPL Mode: Architecture, Transparency & Reproducibility

This document describes the two orchestration modes in the evidence
verification system, their architectural differences, and implications
for academic reproducibility.

---

## Overview

Both modes implement the same **evidence-programming paradigm**: the LLM
generates Python code that calls `evidence_api` functions in a persistent
Jupyter kernel.  They share the same pure-Python library (`evidence_api`,
`subagents`, `classifier`, `compressor`, `data_models`).  The difference
is in **what drives the agent loop**.

```
              ┌──────────────────────────────────────────┐
              │         Pure Python Library               │
              │  evidence_api · subagents · classifier    │
              │  compressor · data_models · renderers     │
              └────────┬──────────────────┬──────────────┘
                       │                  │
          ┌────────────┴───┐    ┌─────────┴─────────────┐
          │  Mode A (SDK)  │    │  Mode B (REPL)        │
          │  Claude Agent  │    │  repl_orchestrator.py  │
          │  SDK + MCP     │    │  OpenAI-compatible     │
          │  nb_execute    │    │  client + code-fence   │
          └────────────────┘    └───────────────────────┘
```

---

## Mode A: SDK mode (`--mode sdk`)

### How it works

The **Claude Agent SDK** (open source: [`anthropics/claude-code-sdk`](https://github.com/anthropics/claude-code-sdk))
spawns the Claude CLI as a subprocess.  The LLM interacts with an MCP server
(`notebook-tools`) via tool calls (`nb_execute`, `nb_markdown`,
`nb_render_papers`, etc.).  Code runs inside a Jupyter kernel managed by the
MCP server.

```
Claude Agent SDK  →  calls nb_execute(code)  →  MCP server  →  Jupyter kernel
                  ←  kernel stdout            ←              ←
                  →  calls nb_render_papers   →  adds cell to .ipynb
```

### Key characteristics

- **Agent loop**: managed by the Claude Agent SDK (open-source Python/TS)
- **LLM protocol**: Anthropic Messages API
- **Tool dispatch**: MCP tool calls (`nb_execute`, `nb_render_*`, etc.)
- **Output**: Jupyter notebook (`.ipynb`) as audit trail
- **Endpoint**: Anthropic-compatible (e.g. `api.z.ai/api/anthropic`)
- **Error handling**: SDK/CLI manages retries, context, tool-error formatting

### Example usage

```bash
# CLI
uv run python -m pkevolve.verification.evidence_programming \
    --claim "Does MAPK1 directly activate H3-3A?" \
    --mode sdk --model glm-5

# With YAML config
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/example_config.yaml --mode sdk
```

---

## Mode B: REPL mode (`--mode repl`)

### How it works

A plain **OpenAI-compatible chat loop** in `repl_orchestrator.py`.  No Claude
SDK, no MCP server.  The script directly manages the conversation, parses
Python code blocks from LLM responses, and executes them in a Jupyter kernel
via `KernelRunner`.

```
Python script  →  LLM.chat.completions.create()  →  parse ```python``` block
               ←  assistant response              ←
               →  KernelRunner.execute(code)      →  Jupyter kernel
               ←  kernel stdout                   ←
               →  feed output as next user msg    →  LLM (next turn)
```

### Key characteristics

- **Agent loop**: explicit Python `for` loop (fully inspectable)
- **LLM protocol**: OpenAI Chat Completions API
- **Tool dispatch**: regex-parsed ` ```python``` ` code blocks
- **Output**: `conversation.json` + `verdict.json`
- **Endpoint**: any OpenAI-compatible endpoint (vLLM, Z.AI, SGLang, etc.)
- **Error handling**: explicit `max_turns` counter + forced verdict

### Example usage

```bash
# CLI
uv run python -m pkevolve.verification.evidence_programming \
    --claim "Does p53 activate BAX?" \
    --mode repl --model glm-5

# With YAML config
uv run python -m pkevolve.verification.evidence_programming \
    --config experiments/example_config.yaml --mode repl
```

---

## Side-by-side comparison

| Aspect               | SDK mode                              | REPL mode                                |
|----------------------|---------------------------------------|------------------------------------------|
| Agent loop           | Claude Agent SDK (open-source)        | Python `for` loop in `repl_orchestrator` |
| LLM protocol         | Anthropic Messages API                | OpenAI Chat Completions API              |
| Tool dispatch        | MCP tool calls                        | Regex-parsed code blocks                 |
| LLM endpoint         | Anthropic-compatible                  | OpenAI-compatible                        |
| Output artifacts     | `.ipynb` notebook                     | `conversation.json` + `verdict.json`     |
| Dependencies         | `claude_agent_sdk`, MCP server        | `openai`, `jupyter_client`               |
| Conversation logging | Streamed messages (loggable)          | Full JSON saved to disk automatically    |
| Error recovery       | SDK-managed retries                   | Manual max-turns + forced verdict        |
| Best for             | Rich audit trails, interactive review | Lightweight experiments, local models    |

---

## Open-source status & transparency

### Claude Agent SDK is fully open source

The [Claude Agent SDK](https://github.com/anthropics/claude-code-sdk) is
open-source.  The `query()` function, message handling, `ClaudeAgentOptions`
construction, and tool dispatch logic are all inspectable Python/TypeScript
code.  This means:

- The **agent loop internals** (retry logic, context management, message
  formatting) are auditable in the SDK source
- The **MCP tool dispatch** follows the open MCP specification
- The **system prompt** you provide via `ClaudeAgentOptions.system_prompt`
  is the one used by the model

### Claude CLI binary: known injections

The SDK spawns the **Claude CLI binary** (`claude`) as a subprocess.  This
binary is also open-source but has some protocol-level behaviors worth noting:

1. **Beta headers**: The CLI sends `anthropic-beta` headers with feature flags
   like `interleaved-thinking-*`, `tool-examples-*`, etc.  The env var
   `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` attempts to suppress these,
   but it is unreliable in some provider configurations (see
   `Claude_CLI_with_non_Anthropic_endpoints.md` for details).

2. **System prompt metadata**: Starting with CLI v2.1.37, the binary may
   inject `x-anthropic-billing-header` metadata into the system prompt body.
   This is generated internally and not strippable via custom headers.

3. **Streaming protocol**: The CLI uses Anthropic-specific SSE events
   (`message_start`, `content_block_delta`, `message_stop`).

**Practical impact when using Z.AI's endpoint**: Z.AI's native
Anthropic-compatible endpoint at `api.z.ai/api/anthropic` handles these
injections gracefully — it ignores unrecognized headers and beta flags.
The billing metadata in the system prompt is treated as inert text by the
model and does not affect generation.  Our env configuration sets
`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` and
`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` as defense-in-depth.

### Version pinning

- The `claude_agent_sdk` Python package version is pinned in `pyproject.toml`
- The CLI binary version should be documented when reporting results
- Check with: `claude --version` (if installed globally) or via SDK logs

---

## Implications for academic publication

### Both modes are suitable for publication

Since the Claude Agent SDK is open source, both modes provide sufficient
transparency for academic reproducibility:

| Transparency concern       | SDK mode                                    | REPL mode                          |
|---------------------------|---------------------------------------------|------------------------------------|
| System prompt inspectable | Yes — set via `system_prompt` param          | Yes — in source code               |
| Agent loop auditable      | Yes — SDK source is open                     | Yes — plain Python loop            |
| Conversation logged       | Yes — via `AssistantMessage`/`ResultMessage`  | Yes — `conversation.json` on disk  |
| API calls inspectable     | Yes — SDK source shows exactly what is sent  | Yes — single `create()` per turn   |
| CLI metadata injection    | Minor — inert text on Z.AI endpoint          | N/A                                |
| Version reproducible      | Pin SDK + CLI version                        | Pin `openai` package version       |

### Recommendations

1. **Report the SDK/CLI version** in your methods section (e.g.
   `claude_agent_sdk==X.Y.Z`, `claude CLI vA.B.C`)
2. **Save the full conversation** — SDK mode can be extended to serialize
   the message stream to JSON (REPL mode already does this)
3. **Save the config YAML** for each run via `cfg.save_yaml()` — this
   captures all parameters except API keys
4. **Document the system prompt template** — it is defined in
   `evidence_programming.py` and fully controllable
5. **Pin the model identifier** in the YAML config for exact reproduction

### When to prefer REPL mode

REPL mode has a slight edge in simplicity for publication:

- **Fewer moving parts** — no MCP server subprocess, no CLI binary
- **Conversation JSON saved by default** — no additional instrumentation
- **Works with any provider** — vLLM, SGLang, OpenAI, local models
- **Simpler dependency chain** — `openai` + `jupyter_client` only

### When to prefer SDK mode

SDK mode is preferable when:

- **Notebook output** is desired for interactive review or supplementary materials
- **Rich tool schemas** are needed (MCP provides typed tool descriptions)
- **Built-in error recovery** from the SDK reduces manual retry logic
- **The audit trail** benefits from interleaved code cells, markdown, and tables

---

## Configuration

Both modes are configured via `pydantic-settings` (`VerificationSettings`)
with a unified priority chain:

```
CLI flags  >  environment variables  >  YAML config file  >  defaults
```

Example YAML config (shared by both modes — `claim` is always passed via
`--claim` on the CLI since it varies per run):

```yaml
mode: repl          # or: sdk
max_iterations: 8
sufficiency_threshold: 0.80

llm:
  model: glm-5
  subagent_model: glm-4.6
  subagent_base_url: "http://localhost:8000/v1/"
  agent_base_url: "https://api.z.ai/api/anthropic"
  temperature: 0.2
```

See `experiments/example_config.yaml` for a complete example.
