# Making Claude CLI work with non-Anthropic API endpoints

Claude Code's binary injects Anthropic-proprietary headers (`anthropic-beta`, `anthropic-version`), beta query parameters (`?beta=true`), and billing metadata into system prompts — all of which cause **400 errors, silent connection drops, or cascading failures** on non-Anthropic backends. The problem is well-documented across dozens of GitHub issues and has spawned an ecosystem of 10+ open-source proxy tools. The most practical solutions, in order of reliability: use a purpose-built translation proxy like **claude-code-router** or **anthropic-proxy**, or connect directly to providers that now expose Anthropic-compatible `/v1/messages` endpoints natively (OpenRouter, Ollama 0.15+, LM Studio, DeepSeek, GLM).

## Exactly what the binary sends that breaks things

Claude Code unconditionally injects several protocol elements that non-Anthropic backends reject. The `anthropic-beta` header carries feature flags like `context-1m-2025-08-07`, `web-search-2025-03-05`, `tool-examples-2025-10-29`, `interleaved-thinking-2025-05-14`, and `oauth-2025-04-20`. Backends that don't recognize these values return **HTTP 400** with messages like `"Unexpected value(s) for the anthropic-beta header"`. The `?beta=true` query parameter gets appended to `/v1/messages` and `/v1/messages/count_tokens` — Ollama, for instance, returns 404 on these endpoints and then enters a degraded state requiring a full restart.

Starting with v2.1.37, Claude Code also injects `x-anthropic-billing-header` metadata directly into the system prompt text body (not as an HTTP header), which AWS Bedrock rejects as a reserved keyword. This is not strippable via `ANTHROPIC_CUSTOM_HEADERS` because it's generated internally. Nearly every major version bump introduces new headers or beta flags — regressions are documented at versions **2.0.31, 2.0.65, 2.1.23, and 2.1.37**. The streaming protocol uses Anthropic-specific SSE events (`message_start`, `content_block_delta`, `message_stop`) that differ from OpenAI's format, and the CLI sends Anthropic-specific tool schemas (`bash_20250124`, `text_editor_20250124`) that foreign backends don't understand.

## Environment variables that help (and their limitations)

Claude Code respects several environment variables for endpoint redirection and feature suppression. The essential trio for any third-party setup:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"    # Redirect all API calls
export ANTHROPIC_AUTH_TOKEN="your-proxy-key"          # Bearer token for auth
export ANTHROPIC_API_KEY=""                            # MUST be empty to prevent Anthropic auth
```

Additional variables reduce problematic traffic:

```bash
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1       # Attempts to suppress beta headers
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1     # Stops telemetry/update pings
export DISABLE_NON_ESSENTIAL_MODEL_CALLS=1             # Reduces unnecessary API calls
```

Model routing uses tier-based overrides: `ANTHROPIC_MODEL` for the default, plus `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, and `ANTHROPIC_DEFAULT_OPUS_MODEL` for slot-specific mapping to non-Anthropic model names.

**Critical caveat**: `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` is **unreliable**. Per GitHub issue #20031, the code only applies this flag when the detected provider is `"anthropic"` or `"foundry"` — when routing through other gateways where the provider is detected differently, the flag is ignored and beta headers still get sent. It also doesn't work when set in `~/.claude/settings.json`'s env block (issue #11960); it must be set in the actual shell environment. And it never strips the `?beta=true` query parameters or the system-prompt billing metadata.

All settings can be persisted in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_AUTH_TOKEN": "your-key",
    "ANTHROPIC_MODEL": "your-model",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1"
  }
}
```

To bypass the Anthropic login prompt entirely when using a third-party backend, edit `~/.claude.json`:

```json
{
  "customApiKeyResponses": { "approved": ["sk-dummy"], "rejected": [] },
  "hasCompletedOnboarding": true
}
```

## Purpose-built proxy tools that translate the protocol

The community has built **10+ open-source proxies** specifically for this problem. These sit between Claude Code and your actual backend, accepting Anthropic-format requests and translating them to OpenAI-compatible format (or stripping incompatible elements before forwarding).

**claude-code-router** (musistudio, **27.8k GitHub stars**) is the most feature-rich option. It's a Node.js router with provider-specific transformers for DeepSeek, OpenRouter, Ollama, Gemini, and GLM. It strips `cache_control` fields, handles tool-call error tolerance, and manages model routing. Activation is a single command:

```bash
npm install -g claude-code-router
eval "$(ccr activate)"   # Sets ANTHROPIC_BASE_URL=http://127.0.0.1:3456 automatically
```

**anthropic-proxy** (maxnowack, 384 stars) is the simplest option — a single `npx` command:

```bash
OPENROUTER_API_KEY=your-key npx anthropic-proxy
ANTHROPIC_BASE_URL=http://0.0.0.0:3000 claude
```

It handles streaming and non-streaming, maps models via `REASONING_MODEL` and `COMPLETION_MODEL` env vars, and targets OpenRouter by default (configurable via `ANTHROPIC_PROXY_BASE_URL`).

**anthropic-proxy-rs** (m0n0x41d) is a high-performance Rust proxy at ~3MB that translates Anthropic requests to OpenAI-compatible format with full streaming SSE support, tool calling, and extended thinking. Works with OpenRouter, OpenAI, Azure, Together.ai, and local LLMs.

**anthropic_adapter** (abhiram1809) is a Python/FastAPI proxy with a clean architecture: `Anthropic Format → Transform → OpenAI Format → Forward → Response → Transform → Anthropic Format`. Configuration:

```bash
# .env
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
OPENAI_API_KEY=your-key

# Usage
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_AUTH_TOKEN="your-openai-key"
export ANTHROPIC_API_KEY=""
```

Other notable projects include **y-router** (Cloudflare Worker deployment), **CCORP** (Rust proxy with web UI for runtime model switching), **anthropic-bridge** (pip-installable with Gemini reasoning cache support), and **LLM-API-Key-Proxy** (universal gateway with key rotation and failover).

## Reverse proxy configs for header stripping only

When your backend already speaks Anthropic format but chokes on specific headers, a lightweight reverse proxy that strips the offending headers is simpler than a full translation proxy.

**nginx** strips headers by setting them to empty:

```nginx
server {
    listen 8080;
    location /v1/messages {
        proxy_set_header anthropic-beta "";
        proxy_set_header anthropic-version "";
        proxy_buffering off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_read_timeout 3600s;
        proxy_pass https://your-backend;
        proxy_ssl_server_name on;
    }
}
```

**Caddy** uses the `-` prefix to delete headers inside `reverse_proxy` blocks:

```caddyfile
:8080 {
    reverse_proxy https://your-backend {
        header_up -anthropic-beta
        header_up -anthropic-version
        transport http {
            tls_server_name your-backend
        }
    }
}
```

**mitmproxy** offers the quickest one-liner for development:

```bash
mitmdump --modify-headers "/~q/anthropic-beta/" --modify-headers "/~q/anthropic-version/"
```

For full request/response body transformation (Anthropic ↔ OpenAI format), nginx's NJS module can handle non-streaming requests, and NGINX has published a [complete working example](https://blog.nginx.org/blog/using-nginx-as-an-ai-proxy) with source code at `github.com/nginx/nginx-demos/tree/main/nginx/ai-proxy`. Streaming SSE transformation is significantly more complex in nginx and better handled by the dedicated proxy tools above.

## Providers with native Anthropic-compatible endpoints

Several providers now expose `/v1/messages` natively, eliminating the need for any translation proxy:

**OpenRouter** offers an "Anthropic Skin" that behaves like the Anthropic API directly:

```bash
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="sk-or-your-key"
export ANTHROPIC_API_KEY=""
```

**Ollama 0.15+** added native Anthropic Messages API support:

```bash
export ANTHROPIC_BASE_URL=http://localhost:11434
export ANTHROPIC_AUTH_TOKEN=ollama
claude --model qwen3-coder
```

**LM Studio 0.4.1+** provides Anthropic-compatible `/v1/messages`:

```bash
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=lmstudio
```

Other providers with native Anthropic endpoints: **DeepSeek** (`https://api.deepseek.com/anthropic`), **GLM/Z.ai** (`https://api.z.ai/api/anthropic`), **MiniMax** (`https://api.minimax.io/anthropic`), **Alibaba DashScope** (`https://dashscope-intl.aliyuncs.com/apps/anthropic`), and **Moonshot/Kimi** (`https://api.moonshot.ai/anthropic/`). These still may not handle all Claude Code's beta headers gracefully — pairing them with `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` is recommended.

## Claude Agent SDK: `query()` vs `ClaudeSDKClient`

The `claude-agent-sdk` Python package (pip: `claude-agent-sdk`) wraps the bundled Claude CLI binary and exposes two distinct APIs for programmatic use. Understanding the differences is critical when connecting to non-Anthropic backends, because one API is substantially more robust for single-prompt agent workflows.

### `query()` — one-shot, fire-and-forget

`query()` is a thin wrapper (~15 lines) around `InternalClient.process_query()`. It spawns the CLI subprocess, writes the prompt to stdin, **immediately closes stdin (EOF)**, and yields streamed `Message` objects. Cleanup happens automatically in a `finally` block.

Key characteristics:
- Stdin closes after prompt delivery → the CLI knows the conversation is complete
- If the backend fails silently (no response, auth error), the CLI process terminates naturally because its input stream is closed
- Entrypoint identifier: `"sdk-py"`
- No session state — each call is independent
- Ideal for agent loops: prompt → tool use → collect results

### `ClaudeSDKClient` — persistent, multi-turn session

`ClaudeSDKClient` (~200 lines) manages a long-lived CLI process. `connect()` spawns the CLI with stdin **kept open** (via an empty async generator that never yields). It supports multi-turn conversations via `.query()`, plus `interrupt()`, `set_model()`, `rewind_files()`, and `get_mcp_status()`.

Key characteristics:
- Stdin stays open indefinitely → the CLI waits for more input
- If the backend fails silently, the CLI process **hangs forever** because it's still waiting for input
- Entrypoint identifier: `"sdk-py-client"`
- Requires explicit `disconnect()` for cleanup
- Timeouts: `CLAUDE_CODE_STREAM_CLOSE_TIMEOUT` (default 60000ms), `initialize_timeout` (60s)
- Ideal for interactive/multi-turn sessions where you need ongoing conversation state

### Practical implications for non-Anthropic backends

When using a non-Anthropic endpoint (e.g., GLM at `api.z.ai`), `query()` is strongly preferred for single-prompt agent workflows. The stdin-closing behavior provides a natural timeout mechanism — if the backend doesn't respond, the process exits rather than hanging indefinitely. `ClaudeSDKClient` should only be used when multi-turn conversation state is genuinely needed, and even then, explicit timeout handling and `disconnect()` calls are essential.

### `ANTHROPIC_API_KEY` vs `ANTHROPIC_AUTH_TOKEN` — a critical distinction

When the CLI detects `ANTHROPIC_API_KEY` in the environment (even as an empty string `""`), it enters an Anthropic-native authentication flow that can stall or fail against non-Anthropic backends. The correct approach for third-party endpoints is:

- Set `ANTHROPIC_AUTH_TOKEN` to your API key (this becomes the `Authorization: Bearer` header)
- **Completely remove** `ANTHROPIC_API_KEY` from the environment — do not set it to empty string
- In Python, use `env.pop("ANTHROPIC_API_KEY", None)` to ensure it's absent

This combination (`ANTHROPIC_AUTH_TOKEN` present, `ANTHROPIC_API_KEY` absent) reliably routes requests to the `ANTHROPIC_BASE_URL` endpoint without triggering Anthropic-specific auth paths.

### Environment setup pattern (Python)

```python
import os
from claude_agent_sdk import query, ClaudeAgentOptions

env = os.environ.copy()
env["ANTHROPIC_BASE_URL"] = "https://api.z.ai/api/anthropic"
env["ANTHROPIC_AUTH_TOKEN"] = os.getenv("GLM_API_KEY")
env["ANTHROPIC_MODEL"] = "glm-4.6"
env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
env.pop("ANTHROPIC_API_KEY", None)  # MUST be absent, not empty

options = ClaudeAgentOptions(
    model="glm-4.6",
    allowed_tools=["mcp__my_server__my_tool"],
    mcp_servers={
        "my-server": {
            "type": "stdio",
            "command": "uv",
            "args": ["run", "my-mcp-server"],
        }
    },
    permission_mode="acceptEdits",
)

# One-shot agent call (preferred for non-Anthropic backends)
for message in query(
    prompt="Your prompt here",
    options=options,
    env=env,
):
    if message.type == "result":
        print(message.result_text)
```

## Conclusion

The root cause is a design tension: Claude Code's binary aggressively enables experimental features via headers and query parameters that **cannot be suppressed** through any official configuration. The `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` flag is buggy and incomplete — it only works for certain detected providers, ignores `settings.json`, and doesn't strip query parameters or system-prompt metadata. Every major Claude Code version risks breaking proxy setups with new headers.

The most robust solution today is a **dedicated translation proxy** (claude-code-router for breadth of provider support, anthropic-proxy for simplicity) combined with the environment variable trio (`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_API_KEY` **removed**). For local development with open models, Ollama 0.15+ and LM Studio 0.4.1+ now work directly. For programmatic Python usage via the Claude Agent SDK, prefer `query()` over `ClaudeSDKClient` when targeting non-Anthropic backends — the stdin-closing behavior prevents silent hangs. Pinning Claude Code versions (using `npm install -g @anthropic-ai/claude-code@2.1.22` to avoid regressions) is a pragmatic defensive measure until Anthropic provides reliable header suppression.