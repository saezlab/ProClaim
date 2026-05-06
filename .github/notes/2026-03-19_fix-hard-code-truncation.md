# Fix Hard-Coded Notebook MCP Truncation — 2026-03-19

**Branch:** `fix/hard-code-truncation`

## Summary

The notebook MCP server (`notebook_mcp.py`) hard-coded aggressive truncation limits (`[:500]` for `nb_execute`, `[:200]` for `nb_render_*`) that left the evidence-programming agent effectively blind to most cell outputs. This change replaces blind slicing with a smart truncation helper that informs the agent when data is clipped, adds a new `nb_read_output` MCP tool so the agent can retrieve full cell outputs on demand, and wires the truncation limit through the existing `max_output_chars` config field.

## Modified Files

| File | Changes |
|---|---|
| `src/proclaim/verification/notebook_mcp.py` | Added `_smart_truncate()` helper with `[...TRUNCATED]` marker; replaced all 6 hard-coded `[:N]` slices; added `nb_read_output` MCP tool; limits now read from `NB_MAX_OUTPUT_CHARS` env var (default 12000) |
| `src/proclaim/verification/config.py` | Updated `max_output_chars` description to reflect it controls all notebook tool limits; value passed as `NB_MAX_OUTPUT_CHARS` env var in `build_sdk_env()` |
| `src/proclaim/verification/evidence_programming.py` | Registered `mcp__notebook-tools__nb_read_output` in `allowed_tools`; added system prompt rule to call `nb_read_output` when output is truncated |

## Key Design Decisions

- **Single config field (`max_output_chars`)** — reuses the existing setting rather than adding per-tool fields, keeping config surface minimal. All three tool types (`nb_execute`, `nb_render_*`, `nb_read_output`) use the same limit.
- **Smart truncation over blind slicing** — `_smart_truncate()` appends a `[...TRUNCATED — showing N of M chars. Call nb_read_output to retrieve the full cell output.]` suffix so the agent always knows when data was clipped and what to do about it.
- **`nb_read_output` as escape hatch** — rather than increasing limits to context-window-busting sizes, the agent gets a lean default and can selectively fetch full output when needed. Reads from the in-memory notebook (already saved to disk), strips HTML tags from `display_data` outputs.
- **Env var passthrough** — the MCP server runs as a subprocess with no direct access to `VerificationSettings`. Config flows via `build_sdk_env()` → `NB_MAX_OUTPUT_CHARS` env var → `os.environ.get()` at import time, matching the existing pattern used for `LLM_BASE_URL`, `LLM_API_KEY`, etc.

## Bug Fixes

- **Agent blindness on `nb_render_*` calls** — renderers use `IPython.display.HTML()` which has no `text/plain` fallback. The previous `[:200]` slice on an empty string meant the agent saw only a brief status message. Now the agent gets up to `max_output_chars` of any available text representation, and can call `nb_read_output` to inspect the HTML-stripped content.
