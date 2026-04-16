# Context Management: Bash + Jupytext with Per-Iteration Context Refresh

Design for an evidence programming agent that uses **bash code
execution** (no Jupyter kernel), **jupytext** for notebook generation,
**EvidenceState** for persistent state, and the **LiteLLM** completion
API (model-agnostic) with per-iteration context refresh.

Inspired by the ACE (Agentic Context Engineering) paradigm.

Supersedes the hybrid recommendation in `repl_vs_directory_ace_analysis.md`,
the passive-notebook position in `REPL_issues.md`, and the prior
Direct-API-with-kernel design in earlier versions of this document.

---

## Problem

The current evidence programming agent uses the Claude Agent SDK, which
accumulates the full conversation history in context: every `nb_execute`
call, its code input, its output, every `nb_markdown` cell, and every
agent reasoning turn.  Over 4–8 iterations of search → extract →
features → sufficiency, this grows to 80–120K tokens.

Cost is proportional to context size × iterations.  Each new agent turn
re-reads the *entire* history.

Additionally, the current system requires a Jupyter kernel managed via
MCP tools (`notebook_mcp.py`, `KernelRunner`), tying code execution
to kernel lifecycle and MCP server lifetime.

---

## Q1: What Does the Original Agent Actually See?

The Claude Agent SDK (`claude_agent_sdk`) is a transparent wrapper
around the Claude Code CLI binary.  The SDK itself does **no context
management** — it spawns `claude --output-format stream-json` as a
subprocess and pipes messages.  All context accumulation happens inside
the CLI process.

### Context flow per turn

| Layer | Content | Size | Persists? |
|-------|---------|------|-----------|
| System prompt | `SYSTEM_PROMPT` with claim, workflow, function docs, schemas | ~4–8K tokens | Every turn |
| Tool definitions | 10 MCP tools (`nb_init`, `nb_execute`, …) + `Read` + `Task` | ~2–3K tokens | Every turn |
| Turn 1 | User prompt ("Verify the following scientific claim…") | ~200 tokens | Until compacted |
| Turn 2 | Assistant → `nb_init` call + result | ~100 tokens | Until compacted |
| Turn 3 | Assistant → `nb_execute` (setup_kernel) + output | ~500 tokens | Until compacted |
| Turn N | Assistant → `nb_execute` (search_pubmed_llm) + **full search output** | 1–12K tokens | Until compacted |
| Turn N+1 | Assistant → `nb_render_papers` + rendered table text | Up to 12K chars | Until compacted |
| Turn N+2 | Assistant → `nb_execute` (extract_and_add_facts) + results | 1–12K tokens | Until compacted |
| Turn N+3 | Assistant → `nb_render_facts` + facts table | Up to 12K chars | Until compacted |
| … | More iterations: search, extract, render, check_sufficiency | 3–12K per tool call | Until compacted |

### Does the model see the notebook file?

**No** — not through the normal MCP flow.  Context flows exclusively through:

1. **System prompt** — static instructions and schemas.
2. **MCP tool return values** — each `nb_execute` / `nb_render_*` call
   returns a text string (truncated at 12K chars via `_smart_truncate`
   in `notebook_mcp.py`).
3. **Built-in `Read` tool** — this is allowed in the tool configuration
   and **could** read the `.ipynb` file if the model chose to, dumping
   the full notebook JSON (metadata, kernelspec, base64 image outputs,
   every cell source and output).  The model is not instructed to do
   this, but it is not prevented.

The model **never** sees kernelspec, notebook metadata, or cell
execution counts through the normal MCP flow.  Since the Jupyter kernel
is persistent, variables survive across `nb_execute` calls — the model
doesn't need prior cell contents for *state*, but it needs the
accumulated conversation messages for *memory* of what it already did.

### Autocompact: the only truncation mechanism

When context approaches the model's window limit (~200K for Sonnet),
the CLI triggers autocompact:

- Summarizes earlier conversation turns into a compacted message
- The summary **replaces** detailed messages — this is lossy
- No CLI flag or API option controls the compaction threshold or
  summary quality
- The application has no visibility into when or how compaction occurs

By iteration 2–3, accumulated tool outputs can reach **50–100K+
tokens**, making autocompact likely.  When it fires, the model may
lose detailed evidence (specific paper titles, fact text, exact
confidence values) that was in earlier tool results.

---

## Q2: Which Version Reduces Cost?

The direct version (bash + jupytext + LiteLLM) is significantly cheaper
due to per-iteration context refresh.

### Per-iteration context size comparison

| Iteration | Original (SDK, cumulative) | Direct (context refresh) | Ratio |
|-----------|---------------------------|--------------------------|-------|
| 1 | ~15K (system + prompt + 3–4 tool calls) | ~15K (same — no refresh yet) | 1× |
| 2 | ~40K (iter 1 turns + iter 2 turns) | ~20K (system + execution log) | 2× |
| 3 | ~70K (all prior turns) | ~25K (system + growing log) | 2.8× |
| 4 | ~100K (all prior turns) | ~30K (system + log) | 3.3× |
| 5 | ~120K+ (approaching autocompact) | ~35K (system + log) | 3.4× |

### Cost drivers

| Factor | Original (SDK) | Direct (LiteLLM) |
|--------|----------------|-------------------|
| Context growth | Monotonic — every tool call/result appends to messages until autocompact | **Bounded** — messages reset at each iteration boundary |
| Per-API-call cost | All prior tool results re-sent on every API call within an iteration | Prior iterations compressed into execution log text |
| Autocompact risk | Lossy summarization when context fills — model may forget evidence details | Not needed; the execution log is the compressed record |
| Cache efficiency | System prompt cacheable; tool results are variable, breaking cache prefixes | System prompt always cacheable; execution log shares growing prefix |
| Render overhead | `nb_render_*` returns full HTML table text re-ingested every turn | No render tools; compact `print()` output only |

### Why the direct version is cheaper

1. **Linear vs quadratic growth.**  The SDK re-sends all prior turns on
   every API call.  A 4-iteration run with 5 API calls per iteration =
   20 API calls.  Call $i$ sends ~$O(i)$ tokens.  Total prompt tokens ~
   $\sum_{i=1}^{20} O(i) \propto O(n^2)$.  The direct version resets at
   iteration boundaries, so the inner-loop calls within iteration $k$
   re-send only the messages from iteration $k$ plus the (growing but
   shared) execution log.

2. **Execution log is smaller than conversation history.**  The
   conversation includes the assistant's reasoning text, tool call
   JSON, and tool result JSON — with framing overhead.  The execution
   log contains only code and printed output with `# →` prefixed
   comments.  For equivalent content, the log is ~30–50% smaller.

3. **No render overhead.**  The original version's `nb_render_papers`,
   `nb_render_facts`, `nb_render_sufficiency` tools return HTML-to-text
   tables (~2–5K tokens each) into the conversation context.  These are
   visually redundant with the `nb_execute` outputs the model already
   saw.  The direct version has no render tools — output is via
   `print()` in bash.

### Estimated savings

For a typical 4-iteration, 8-iteration-cap verification:

| Metric | Original (SDK) | Direct (LiteLLM) | Savings |
|--------|----------------|-------------------|---------|
| Total prompt tokens | ~400–500K | ~150–200K | ~60% |
| Total completion tokens | ~30–40K | ~30–40K | ~0% (same work) |
| API calls | ~20–30 | ~20–30 | ~0% (same structure) |
| Cache hit rate | Low (variable tool results break prefix) | High (stable system prompt + log prefix) | Significant |

---

## Proposal

**Drop the Jupyter kernel and MCP notebook tools entirely.**  The agent
executes Python code via a bash session, and the execution log is
recorded in jupytext format for post-hoc notebook generation.

At the start of each evidence iteration, **reset the agent context** and
inject two components:

1. **Cached prefix** — system prompt, tool definitions, instructions,
   claim.  Static across iterations.  Eligible for prompt caching
   (Anthropic cache read pricing ≈ free).

2. **Current execution log** — all Python code blocks and their stdout,
   plus agent markdown notes, serialized as text from the jupytext log.
   This is the agent's working memory.

The agent does **not** see its own prior conversation turns.  It sees
the execution log it has been building, which contains richer grounding:
actual Python calls, printed outputs, and its own reasoning notes.

---

## Cost Model

| Segment | Tokens | Cached? | Per-iteration cost |
|---------|--------|---------|-------------------|
| System prompt + tool defs + instructions + claim | ~4–6K | Yes (after first call) | ≈ 0 (cache read) |
| Execution log (grows per iteration) | ~5–25K | Partially (prefix-stable) | Input rate on delta |
| Agent output (new code + reasoning) | ~2–4K | No | Output rate |

**Current (cumulative history):** iteration N pays for all N prior
turns.  Tokens grow quadratically with iterations.

**Proposed (execution log injection):** iteration N pays for the log
size at iteration N.  Tokens grow linearly with iterations.

Estimated 4–5× context reduction at iteration 4.

---

## Why the Execution Log Is Sufficient Context

### Search history

The execution log entry:

```python
# %%
pmids = search_pubmed_llm(state.claim, state, llm)
# → Generated query: "MAPK1 AND phosphorylation AND H3" → 8 new papers
```

is a richer search log than any structured `search_log` field.  It shows
the function called, the query generated, the result count.  The agent
reads this at the start of the next iteration and knows not to re-run it.

### Gap search attempts

Similarly, gap-targeted searches appear as log entries:

```python
# %%
search_for_gap("kinase substrate specificity for H3-3A", state)
# → 3 new papers added
```

No separate `gap_search_attempts` state field needed.

### Reasoning continuity

The agent writes markdown blocks in the execution log explaining its
reasoning between steps.  These survive context refresh and serve as
the reasoning trace.

### Sufficiency trajectory

`get_sufficiency_history(state)` prints the confidence trend and returns
`{"trend": "improving" | "flat" | "declining"}`.  Together with
`get_evidence_summary(state)`, the agent can reconstruct the full
status from `EvidenceState` alone.

---

## Safety: Why Re-execution Is Not a Problem

`EvidenceState` has dedup guards at every mutation point:

| Operation | Guard |
|-----------|-------|
| Paper addition | Dedup by PMID and DOI (`_s2_paper_to_record`, `search_pubmed`) |
| Fact extraction | `extracted_pmids` list skips already-processed papers |
| Fact addition | Dedup by `(text.lower(), source_pmid)` |
| Feature population | `populate_paper_features` is idempotent (skips papers with existing features) |
| Sufficiency check | Appends to `sufficiency_history` — safe to re-run (returns new result) |

Even if the agent *does* re-issue a call, the data layer is protected.
The only waste is LLM cost for `search_pubmed_llm` (1 LLM call to
generate a query) — but since the agent can see its prior search entry
in the execution log, it should not re-issue it.

---

## ACE Analogy

ACE's Generator receives:

    playbook (evolving artifact) + current question → answer

Our proposed agent receives:

    execution log (evolving artifact) + continuation prompt → next iteration

Key parallel: the agent does **not** need its prior conversation turns.
It needs the *artifact it has been building*.

### Difference from ACE

ACE is multi-episode: the playbook improves across tasks.  Our notebook
is single-episode: it tracks one claim's verification.  But within a
single verification, the dynamics are the same — state accumulates in a
persistent artifact, and the LLM context resets with that artifact
injected.

ACE is also stateful due to the Reflector: each round's reflection feeds
back into the Curator, which updates the playbook.  Our sufficiency
classifier is the analogous Reflector — it produces structured feedback
(gaps, confidence, trend) that the agent uses to decide the next search
action.

---

## Execution Log Growth Management

A typical 4-iteration verification produces ~15–20 code blocks
(~15–25K tokens).  At the 8-iteration cap, this may reach ~40–50K
tokens.

### Mitigation (if needed)

1. **Truncate early cell stdout.**  Keep the code (the search log) but
   trim stdout for blocks older than N iterations.  The code line
   `search_pubmed_llm(state.claim, state, llm)` tells the agent what
   was done; the full paper listing is less useful 3 iterations later.

2. **Sliding window + summary.**  Inject `get_evidence_summary()` output
   as a preamble, then only the last N iterations' blocks verbatim.

3. **State summary replaces log.**  After sufficient iterations, replace
   the full execution log with a structured summary from
   `get_evidence_summary()` + `get_sufficiency_history()`.

Option 1 is the natural starting point.  Options 2–3 are fallbacks if
high-iteration claims hit context limits.

---

## Architecture: Bash + Jupytext + LiteLLM Direct API

### What Changes from Current System

| Component | Current | Proposed |
|-----------|---------|----------|
| Code execution | `nb_execute` → MCP → KernelRunner → jupyter_client | Bash → `python3 -c "..."` |
| State persistence | EvidenceState + notebook cells in memory | EvidenceState on disk (auto-save) |
| Audit trail | Live `.ipynb` via nbformat | Post-hoc `.ipynb` via jupytext |
| Agent tools | 9 custom MCP tools (`nb_*`) | `bash` + `read_file` |
| Context refresh | None (history accumulates) | Per-iteration (rebuild messages) |
| LLM provider | Anthropic only (via Claude Agent SDK) | Any (via LiteLLM: Anthropic, OpenAI, vLLM, etc.) |
| Heavy deps | nbformat, nbclient, jupyter_client, ipykernel, FastMCP | jupytext, litellm |

### Key Insight

EvidenceState auto-saves to `evidence_state.json` after every mutation
(`add_paper`, `add_fact`, `add_conflict`).  There is **no in-memory
state** that requires a persistent kernel.  Each Python invocation
loads state from disk, does work, and state auto-saves.  The kernel
was only needed because MCP tools executed code in a kernel.  Without
MCP, the kernel is unnecessary.

Heavy ML model loading (SBERT, NLI, MLP) only occurs in
`populate_paper_features()`, which runs once per iteration as a batch.
Not a per-command concern.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Outer Orchestrator (Python)                            │
│                                                         │
│  ┌────────────────┐    ┌─────────────────────────────┐  │
│  │ Bash session   │    │ LiteLLM                     │  │
│  │ (subprocess)   │    │ litellm.completion()        │  │
│  └──────┬─────────┘    └──────────┬──────────────────┘  │
│         │                         │                     │
│         │    ┌────────────────────┤                     │
│         │    │  Tool-use loop     │                     │
│         │    │  (build messages,  │                     │
│         │    │   dispatch tools,  │                     │
│         │    │   refresh context) │                     │
│         │    └────────────────────┘                     │
│         │                                               │
│  ┌──────┴────────────────────────────────────────────┐  │
│  │ Tool dispatch:                                    │  │
│  │   bash(code) → python3 -c "..." → stdout          │  │
│  │   read_file(path) → file contents                 │  │
│  │ Jupytext logger: appends code+output to .py log   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ evidence_state.json  (auto-saved by evidence_api) │  │
│  │ execution_log.py     (jupytext percent format)    │  │
│  │ evidence_report.ipynb (generated at end)           │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Core Loop

```python
import litellm
import subprocess
from pathlib import Path
from pkevolve.verification.evidence_state import EvidenceState

tools = build_tool_schemas()  # bash + read_file (OpenAI function format)

log_path = workspace / "execution_log.py"
init_jupytext_log(log_path, claim)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": initial_prompt},
]

for iteration in range(1, max_iterations + 1):
    # Inner tool-use loop: runs until agent finishes one iteration
    while True:
        response = litellm.completion(
            model=model,                # e.g. "anthropic/claude-sonnet-4-20250514"
            messages=messages,
            tools=tools,
            max_tokens=16384,
        )
        choice = response.choices[0]
        messages.append(choice.message.model_dump())

        if choice.finish_reason == "stop":
            break

        # Dispatch tool calls
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                result = dispatch_tool(tc.function.name, json.loads(tc.function.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    # Iteration boundary — check stopping condition
    state = EvidenceState.load(workspace / "evidence_state.json")
    if should_stop(state):
        break

    # *** CONTEXT REFRESH ***
    execution_log = log_path.read_text()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"<execution_log>\n{execution_log}\n</execution_log>\n\n"
            f"Continue from iteration {iteration + 1}. "
            f"The execution log above shows all prior work. "
            f"Call check_sufficiency and address remaining gaps."
        )},
    ]

# Generate final notebook
generate_notebook(log_path, workspace / "evidence_report.ipynb")
```

### Model Configuration

LiteLLM uses a provider-prefixed model string.  All LLM providers that
support tool-use / function calling are compatible:

| Provider | Model string example | Notes |
|----------|---------------------|-------|
| Anthropic | `anthropic/claude-sonnet-4-20250514` | Prompt caching via `cache_control` |
| OpenAI | `openai/gpt-4o` | Native function calling |
| vLLM (self-hosted) | `openai/my-model` + `api_base` | Cost = 0 (own GPU) |
| DeepSeek | `deepseek/deepseek-chat` | Low cost |
| Google | `gemini/gemini-2.0-flash` | Function calling support |

Switch models by changing a single config string.  The outer loop,
tool dispatch, and context refresh are model-agnostic.

### Tool Dispatch

Only two tools: `bash` (execute Python code) and `read_file`.  The
bash tool runs Python code, captures stdout/stderr, and appends the
code + output to the jupytext log:

```python
def dispatch_tool(name: str, arguments: dict) -> str:
    if name == "bash":
        code = arguments["command"]
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=300,
            cwd=str(workspace),
        )
        output = result.stdout + result.stderr
        append_to_jupytext_log(log_path, code, output)
        return _smart_truncate(output)
    elif name == "read_file":
        return Path(arguments["path"]).read_text()
```

### Tool Schema (OpenAI Function Calling Format)

LiteLLM uses the OpenAI function calling format, which is normalized
across all providers:

```python
def build_tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a Python command. Use python3 -c '...' style.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    ]
```

### Jupytext Format

The execution log is recorded in jupytext **percent format** — a plain
`.py` file that jupytext converts to `.ipynb`:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
# ---

# %% [markdown]
# # Evidence Report: MAPK1 phosphorylates H3
# Claim verification started at 2026-04-11T10:00:00

# %% [markdown]
# ## Iteration 1: Initial Search

# %%
from pkevolve.verification.evidence_api import *
from pkevolve.verification.evidence_state import EvidenceState
state = EvidenceState.load("evidence_state.json")
llm = make_llm(...)
pmids = search_pubmed_llm(state.claim, state, llm)
# → Generated query: "MAPK1 AND phosphorylation AND H3" → 8 new papers

# %%
for pmid in list(state.papers.keys())[:5]:
    extract_and_add_facts(llm, pmid, state)
# → Extracted 3 facts from 5 papers

# %%
result = check_sufficiency(state, llm)
print(result)
# → SufficiencyResult(label='INSUFFICIENT', confidence=0.42, gaps=[...])
```

At the end of verification: `jupytext --to notebook execution_log.py`
produces `evidence_report.ipynb`.

### Prompt Caching

The system prompt is static across iterations and eligible for caching.
Anthropic models support explicit cache breakpoints via `cache_control`;
LiteLLM passes these through transparently.  For other providers,
the system message is re-sent each iteration but the per-iteration
cost is linear (not quadratic) because the execution log replaces
cumulative history.

### Token Accounting

LiteLLM normalizes `response.usage` across providers:

```python
response.usage.prompt_tokens         # input tokens
response.usage.completion_tokens     # output tokens
response.usage.total_tokens          # sum
```

For Anthropic models, cache-specific fields are also available:

```python
response.usage.cache_creation_input_tokens  # first-call cache write
response.usage.cache_read_input_tokens      # cache hits
```

Exact per-call token accounting is essential for the paper's cost
analysis section.

### Iteration Boundary Detection

The agent signals iteration completion by calling `check_sufficiency()`
(which appends to `sufficiency_history` and auto-saves state).  The
outer loop detects this by monitoring `evidence_state.json`:

```python
def should_stop(state: EvidenceState) -> bool:
    if (workspace / "verdict.json").exists():
        return True
    if not state.sufficiency_history:
        return False
    last = state.sufficiency_history[-1]
    return last["confidence"] >= threshold
```

The inner tool-use loop runs until `finish_reason == "stop"` — the LLM
decides when an iteration is complete (after searching, extracting,
checking sufficiency).  The outer loop then refreshes context and sends
a continuation prompt.

---

## SDK Entry Point Comparison (for NeurIPS)

This system is intended for a NeurIPS submission.  The choice of API
entry point affects reproducibility, describability, and evaluation
precision.

| Criterion | `query()` per iteration | `ClaudeSDKClient` | **Direct API (LiteLLM)** |
|-----------|------------------------|--------------------|------------------------|
| Context refresh | ✓ Fresh each call | ✗ History accumulates | **✓ Full control** |
| Describable in paper | One-sentence ("invoke agent per iteration") | Complex (adaptive reset) | **Algorithm 1 IS the code** |
| Token accounting | Coarse (`ResultMessage.usage`) | `get_context_usage()` aggregated | **Exact per-call `response.usage`** |
| Prompt caching control | Via SDK (limited) | Via SDK (limited) | **Explicit (Anthropic); provider-dependent** |
| Model flexibility | Anthropic only (Claude Code CLI) | Anthropic only | **Any provider (Anthropic, OpenAI, vLLM, etc.)** |
| Reproducibility | Requires 231 MB CLI binary | Same | **`pip install litellm jupytext`** |
| Bash tool | Built-in (well-tested) | Built-in | Implement (`subprocess.run`, ~10 lines) |
| Tool dispatch | SDK-managed (opaque) | SDK-managed | Self-managed (~30 lines) |
| Edge cases (retries, rate limits) | SDK handles | SDK handles | LiteLLM handles (built-in retries, fallbacks) |
| CLI binary dependency | Yes (opaque ELF) | Yes | **None** |

### Recommendation: Direct API via LiteLLM

**For the paper:** The algorithm described in the paper IS the
implementation.  No "we use SDK X which internally does Y" indirection.
A reviewer can read Algorithm 1 and understand exactly what context the
agent sees at each iteration.

**For evaluation:** Exact token counts per API call enable precise cost
analysis.  LiteLLM normalizes `response.usage` across providers.  For
Anthropic, cache hit rates are directly observable.  These metrics are
needed for the paper's cost comparison with baselines.

**For model-agnostic evaluation:** LiteLLM lets us run the same
algorithm across Claude, GPT-4o, DeepSeek, or self-hosted vLLM with
a single config change.  This enables model ablation experiments and
strengthens the paper's generality claim.

**For reproducibility:** A reader needs only `pip install litellm
jupytext` and the paper's code.  No version-specific CLI binary.

**For rapid prototyping:** If time-constrained, implement with `query()`
first (it works with the same outer loop structure), then migrate to
Direct API before submission.  The outer loop (iteration, context
refresh, stopping condition) is identical.

### Why Not `query()` per iteration

`query()` works with the bash + jupytext approach (no kernel to kill).
Each `query()` call is a fresh context.  The agent uses the built-in
`Bash` and `Read` tools.  It is the simplest to implement.

Downsides for a NeurIPS paper:

1. **Opaque context management.**  The SDK builds its own system prompt,
   prepends tool definitions, and may add internal instructions.  You
   don't know exactly what the agent sees.

2. **Imprecise token accounting.**  `ResultMessage.usage` gives totals,
   not per-call breakdowns.  Can't compute cache hit rates.

3. **Reproducibility risk.**  The Claude Code CLI binary is version-
   specific.  Behavior may change between `v0.1.58` and a future release.
   A paper should not depend on an opaque binary for its core algorithm.

4. **Cannot describe algorithmically.**  The paper would say "we call
   `query(prompt, options)` which internally manages tool dispatch" —
   this is not an algorithm, it's an API call.

### Why Not `ClaudeSDKClient` session

Context accumulates across `query()` calls within a session.  This is
the original problem the design solves.  Adaptive reset (disconnect +
reconnect at pressure threshold) is possible but:

- Adds complexity without eliminating the fundamental issue
- `get_context_usage()` monitoring is coarse (no per-call breakdown)
- Disconnect/reconnect restarts the CLI subprocess
- Not needed: bash sessions don't require persistence

---

## Why Code Execution, Not MCP Tools

A reviewer might ask: why not expose the evidence API functions as MCP
tools (or Anthropic tool-use functions) instead of using code execution?

MCP tools are a **fixed vocabulary**.  Code execution is a **generative
language**.  Observed agent behavior in real verifications includes:

1. **Ad-hoc filtering:** `[pmid for pmid, count in results.items() if count == 0]`
   — the filter condition varies per claim.

2. **Regex search over paper text:**
   `re.finditer(r'AC1|ADCY1|adenylyl cyclase.*inhibit', full_text)` —
   the agent invents patterns based on evolving understanding.

3. **Direct LLM probing:** `llm("What is the relationship between
   GNAO1 and ADCY1?")` — using the LLM as a reasoning scratchpad.

4. **Extraction debugging:** manually calling `extract_facts()` on a
   specific excerpt to diagnose why extraction returned 0 facts.

5. **Custom prompt engineering:** writing a custom extraction prompt
   to test a hypothesis about the extractor's behavior.

6. **State introspection:** `for pmid, paper in state.papers.items():`
   with custom field selection and truncation.

These are compositional, ad-hoc, claim-specific.  No fixed tool menu
can anticipate them.  A `run_code` MCP tool would be code execution
with extra indirection.

The bash approach strengthens this argument: the agent's tool vocabulary
is `bash` (Python via `python3 -c`) and `read_file` — the most general
possible interface.  All evidence API functions are available as Python
imports, not as tool definitions that the agent must learn.

---

## Utility: Jupytext Log Management

### `init_jupytext_log(path, claim)`

Create the initial `.py` file with jupytext percent format header:

```python
def init_jupytext_log(path: Path, claim: str):
    header = textwrap.dedent(f"""\
        # ---
        # jupyter:
        #   jupytext:
        #     text_representation:
        #       format_name: percent
        # ---

        # %% [markdown]
        # # Evidence Report: {claim}
        # Started: {datetime.now().isoformat()}
    """)
    path.write_text(header)
```

### `append_to_jupytext_log(path, code, output)`

Append a code block and its output as a percent-format cell:

```python
def append_to_jupytext_log(path: Path, code: str, output: str):
    with open(path, "a") as f:
        f.write("\n# %%\n")
        f.write(code)
        if output.strip():
            # Embed output as comments (preserved in notebook conversion)
            for line in output.strip().splitlines():
                f.write(f"\n# → {line}")
        f.write("\n")
```

### `generate_notebook(log_path, notebook_path)`

Convert the jupytext log to `.ipynb`:

```python
def generate_notebook(log_path: Path, notebook_path: Path):
    subprocess.run(
        ["jupytext", "--to", "notebook", "-o", str(notebook_path), str(log_path)],
        check=True,
    )
```

### `serialize_execution_log(path, max_tokens=None)`

Read the log file for context injection.  Optionally truncate old
cell outputs (keep code, trim stdout) if over token budget:

```python
def serialize_execution_log(path: Path, max_tokens: int = 30000) -> str:
    text = path.read_text()
    approx_tokens = len(text) // 4
    if approx_tokens <= max_tokens:
        return text
    # Truncation: keep all code lines, trim output lines from early cells
    return _truncate_old_outputs(text, max_tokens)
```

---

## Changes Required

### New dependencies

Add to `pyproject.toml`:
- `litellm` — model-agnostic LLM completion API (Anthropic, OpenAI, vLLM, etc.)
- `jupytext` — notebook generation from percent-format `.py` files

### Remove dependencies

No longer needed (can be kept for other uses):
- `nbclient`, `jupyter_client`, `ipykernel` — Jupyter kernel stack
- `fastmcp` — MCP server framework
- `claude-agent-sdk` — Claude Code CLI wrapper

### `evidence_programming.py`

- Replace `claude_agent_sdk` import with `litellm`
- Replace the single `async for message in query(...)` loop with the
  outer iteration loop + inner tool-use loop
- Remove `KernelRunner` creation and MCP subprocess management
- Add `dispatch_tool()` for bash + read_file routing
- Add `build_tool_schemas()` for OpenAI function calling format
- Add jupytext log management functions
- Add `should_stop()` based on `evidence_state.json`

### `notebook_mcp.py`

Can be removed or kept for backward compatibility.  No longer used by
the main verification flow.

### `kernel_runner.py`

No longer used by the main verification flow.  Can be kept for testing
or interactive use.

### `EvidenceState`

No changes needed.  Auto-save to disk is the persistence mechanism.

### System prompt

Update to reference `bash` tool instead of `nb_execute`, `nb_markdown`,
etc.  Remove notebook-specific instructions (cell numbering, render
tools).  Add instructions for writing Python code that imports from
`pkevolve.verification.evidence_api`.

---

## What This Does NOT Change

- **Subagent costs** — fact extraction (`extract_and_add_facts`), gap
  identification (`identify_gaps`), query formulation
  (`formulate_gap_queries`) use the subagent LLM.  These costs are
  per-paper/per-gap, independent of outer context size.

- **MLP classifier** — runs locally, no token cost.

- **Search APIs** — PubMed, Semantic Scholar.  No token cost.

- **Notebook as audit trail** — the notebook continues to be produced
  as `.ipynb` for human review.  Generated post-hoc from the jupytext
  execution log rather than maintained live.

- **Evidence API** — all functions in `evidence_api.py` remain the same.
  The agent imports and calls them as Python code, exactly as before.

---

## Alternatives Considered

### Direct API + Jupyter kernel (prior version of this document)

Call Anthropic Messages API directly with a Python-owned `KernelRunner`
for code execution.  **Rejected in favor of bash approach:** the kernel
adds complexity (lifecycle management, jupyter_client dependency) without
benefit.  EvidenceState auto-saves to disk; no in-memory state requires
kernel persistence.  Bash execution is simpler and eliminates 4
dependencies.

### `ClaudeSDKClient` with adaptive reset

Monitor context pressure via `get_context_usage()`, disconnect +
reconnect when threshold exceeded.  **Rejected:** couples context
management to subprocess lifecycle; adds complexity without eliminating
the fundamental problem; imprecise token accounting.

### Per-iteration `query()` calls with MCP notebook tools

One `query()` call per iteration with notebook injected as prompt.
**Rejected:** each call spawns a new CLI subprocess, restarting the
MCP server and Jupyter kernel.  Warm-up overhead every iteration.

### Per-iteration `query()` calls with bash (viable fallback)

Same outer loop structure, but using `query()` instead of Direct API.
`allowed_tools=["Bash", "Read"]`.  **Not chosen but viable:** works
for rapid prototyping.  Migrate to Direct API for final paper
submission to get algorithm transparency and precise token accounting.

### Auto-compaction with low threshold

Use Claude Code CLI's auto-compact feature.  **Rejected:** no CLI flag
or settings.json option to control the compaction threshold.  Hooks are
observation-only.  Compaction summary is uncontrolled.

---

## References

- `doc/repl_vs_directory_ace_analysis.md` — ACE alignment analysis
- `doc/REPL_issues.md` — prior analysis of notebook as passive view layer
- `doc/sdk_vs_repl_modes.md` — SDK vs REPL architecture
- Zhang et al. "Agentic Context Engineering." ICLR 2026.
