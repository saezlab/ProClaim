# RLM REPL Implementation — 2026-02-23

## Branch

`RLM`

## Summary

Implemented the Recursive Language Model (RLM) REPL paradigm for the evidence programming verification subsystem. All 13 MCP evidence tools were extracted into a pure Python library (`evidence_api.py`) that operates on in-memory `EvidenceState` objects. Subagent logic was rewritten as plain functions with injected LLM callables. A shared `KernelRunner` manages Jupyter kernel lifecycle for both orchestration modes. A standalone REPL orchestrator (`repl_orchestrator.py`) enables end-to-end claim verification using any OpenAI-compatible endpoint without the Claude Agent SDK. The demo script now supports `--mode sdk|repl` and uses only the notebook-tools MCP server (the evidence-tools MCP server is no longer needed). Existing MCP wrappers were preserved as thin delegation layers for backward compatibility.

## New Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/proclaim/verification/evidence_api.py` | 683 | Pure Python library: all 13 evidence functions operating on in-memory `EvidenceState`. Includes PubMed search, fact extraction, synthesis, conflict tracking, sufficiency checking, compression, and verdict emission. Contains shared query helpers (`_generate_tiered_queries`, `_extract_symbol_subtokens`, `_BIO_VERB_TO_NOUN`) moved from the former `mcp_tools.py`. |
| `src/proclaim/verification/kernel_runner.py` | 296 | Jupyter kernel lifecycle management shared by both orchestration modes. `KernelRunner` class with `start()`, `execute()`, `inject_prelude()`, `shutdown()`. `outputs_to_text()` helper converts kernel outputs to plain text for LLM feedback. `atexit`-based cleanup via global `_kernels` registry. |
| `src/proclaim/verification/subagents.py` | 333 | Subagent logic as plain Python functions with `LLMCallable = Callable[[str], str]` dependency injection. Functions: `extract_facts()`, `synthesize_subclaim()`, `detect_conflicts()`, `formulate_gap_queries()`. Each constructs a prompt, calls the injected LLM, and parses JSON output. |
| `src/proclaim/verification/repl_orchestrator.py` | 369 | Mode B standalone REPL orchestrator. `verify_claim_repl()` runs a turn-based loop: LLM generates Python code → `extract_code()` parses ` ```python ` fences → kernel executes → output fed back. Terminates when `verdict.json` is written or `MAX_TURNS` reached. Saves `conversation.json` for debugging. `verify_claims_batch()` for sequential processing. |

## Modified Files

| File | Lines | Changes |
|------|-------|---------|
| `src/proclaim/verification/evidence_state.py` | 204 | Added `MAX_ITERATIONS = 8` class variable, `trace: list[dict]` field (excluded from JSON serialization), `append_trace()` method, `checkpoint_save()` method that writes both `evidence_state.json` and `trace.json`. Made `checkpoint_save` accept `str` or `Path`. |
| `src/proclaim/verification/mcp_tools.py` | 353 | Refactored from ~800 lines of inline logic to thin MCP wrappers. Each `@mcp.tool()` body is now 3–5 lines: `_load_state → api.function(state) → _save_state → _trace → return`. All query helpers, PubMed/PMC logic, and classifier/compressor instances removed (now in `evidence_api.py`). |
| `src/proclaim/verification/notebook_mcp.py` | 244 | Replaced inline kernel management (`_check_kernel_available`, `_get_kernel`, `_shutdown_kernels`) with `KernelRunner` delegation. Added `_runner_outputs_to_nb()` to convert KernelRunner output format (`{"type":...}`) to nbformat (`{"output_type":...}`). |
| `src/proclaim/verification/renderers.py` | 460 | Added three `_from_state` variants: `render_papers_from_state()`, `render_facts_from_state()`, `render_sufficiency_from_state()`. These accept an `EvidenceState` object (or plain dict) directly, avoiding disk I/O. Used by the kernel prelude in REPL mode. |
| `scripts/verification/demo_evidence_programming.py` | 436 | Added `--mode sdk\|repl` CLI flag (default: `sdk`). Mode A uses only notebook-tools MCP server; system prompt includes Python setup code for `nb_execute`. Mode B delegates to `repl_orchestrator.verify_claim_repl()`. Removed evidence-tools MCP server entirely. |
| `doc/implementation_plan.md` | — | Retitled to "Evidence Programming via RLM REPL". Rewrote architecture overview, sections 1 (project structure), 3 (evidence API), 4 (subagents), 5 (dual-mode orchestrator), 8 (implementation stages), 9 (dependencies), 10 (design rationale). Stages 1–3 marked DONE. |

## Architecture

```
                 ┌──────────────────────────────────────────┐
                 │         Pure Python Library               │
                 │  evidence_api · subagents · classifier    │
                 │  compressor · data_models · renderers     │
                 └────────┬──────────────────┬──────────────┘
                          │                  │
             ┌────────────┴───┐    ┌─────────┴─────────────┐
             │  Mode A (SDK)  │    │  Mode B (Standalone)   │
             │  Claude Agent  │    │  repl_orchestrator.py  │
             │  SDK + nb_exe  │    │  OpenAI-compatible     │
             │  cute as tool  │    │  client + code-fence   │
             └──────┬─────┬──┘    └────┬────────────┬──────┘
                    │     │            │            │
              ┌─────┘  ┌──┘       ┌────┘       ┌────┘
              ▼        ▼          ▼            ▼
        ┌──────────┐ ┌─────────────────────────────────┐
        │ Notebook  │ │       Jupyter Kernel             │
        │ MCP tools │ │  (KernelRunner)                  │
        │ nb_execute│ │  state = EvidenceState(...)      │
        └───────┬──┘ │  search_pubmed(q, state)         │
                │    │  check_sufficiency(state)        │
          ┌─────┘    │  emit_verdict(...)               │
          ▼          └──────────────────────────────────┘
   ┌───────────────┐
   │  .ipynb file   │
   │  (audit trail) │
   └───────────────┘
```

**Key flow**: The LLM generates Python code. In Mode A, this code is passed to `nb_execute` (a notebook MCP tool). In Mode B, code fences are parsed from the LLM response and sent directly to the `KernelRunner`. Both paths execute in the same Jupyter kernel where `state` is a live Python variable.

## Key Design Decisions

- **State by reference, not disk I/O**: `evidence_api` functions take `state: EvidenceState` and mutate it in-place. No `load_state/save_state` per call — this is the core RLM advantage over MCP tool calls. Disk persistence is deferred to `checkpoint_save()` or the MCP wrappers.

- **`print()` for feedback**: Evidence API functions use `print()` for output. The kernel captures stdout and feeds it back to the LLM as the next user message. This replaces MCP tool return values.

- **`LLMCallable = Callable[[str], str]` for subagents**: Dependency injection makes subagent functions testable with mock LLMs and portable across providers. No Claude Agent SDK `Task` dependency.

- **Jupyter kernel over `exec()`**: Process isolation (kernel crash ≠ orchestrator crash), persistent variables across cells, rich `display_data` output for HTML renderers, and automatic `.ipynb` audit trail.

- **Single MCP server in Mode A**: The evidence-tools MCP server is eliminated. The LLM calls evidence functions directly via `nb_execute` code cells. Only notebook-tools remains for notebook I/O.

- **Thin MCP wrappers preserved**: `mcp_tools.py` retains all 13 `@mcp.tool()` signatures for backward compatibility with the original `orchestrator.py`. Each wrapper delegates to `evidence_api`.

- **`MaxIterationsExceeded` carries state**: The exception class stores the `EvidenceState` object so the orchestrator can emit a verdict from the partially-complete state.

- **`checkpoint_save` accepts `str` or `Path`**: REPL-generated code may pass workspace paths as strings; defensive coercion avoids `AttributeError`.

## Bug Fixes

- **Missing `render_*_from_state` functions**: The kernel prelude imported `render_papers_from_state`, `render_facts_from_state`, `render_sufficiency_from_state` from `renderers.py`, but these variants (operating on in-memory state instead of disk) did not exist. Added all three.

- **`checkpoint_save` type rigidity**: `EvidenceState.checkpoint_save()` required a `Path` object but REPL code could pass a `str`. Fixed to accept both via `Path(workspace)` coercion.

- **`MaxIterationsExceeded` missing `state` attribute**: The exception class was a bare `pass` body with no stored state. Added `state` keyword argument to `__init__` and updated the raise site in `check_sufficiency()` to pass `state=state`.
