

 ## Current Role of the Notebook

The notebook is a **passive documentation artifact** — a read-only evidence report. It never drives the verification loop. The separation works like this:

| Concern | Where it happens |
|---------|-----------------|
| Evidence retrieval, extraction, state mutation | Evidence MCP server (`mcp_tools.py`) |
| Sufficiency checking, verdict emission | Evidence MCP server |
| Rendering state snapshots as HTML tables/cards | Notebook MCP server (notebook_mcp.py) |
| Narrative explanations between steps | Notebook MCP server (`nb_markdown`) |

The `nb_render_*` tools generate code cells like `render_papers(workspace)` that **read** `evidence_state.json` and produce HTML — they never mutate it. The notebook is a log of what happened, not an active participant.

## What Would Happen If Both Were Combined

If the agent could invoke evidence-state tools *from inside* notebook cells (via `nb_execute`), several things would change:

**1. Race conditions on state.** The evidence MCP server runs as a subprocess with its own process. The notebook kernel is a *separate* subprocess. Both would read/write `evidence_state.json` — no locking exists. Concurrent writes would corrupt state.

**2. Loss of the audit trail.** The evidence MCP server traces every tool call to `trace.jsonl` via `_trace()`. Code executed inside notebook cells bypasses this — you'd lose the structured audit log that records which tool was called, with what arguments, and what changed.

**3. Blurred separation of concerns.** Right now the orchestrator agent's decision loop is cleanly visible in the MCP tool call sequence (search → extract → check → gap-search → verdict). If evidence mutations also happened inside notebook cells, the agent's reasoning would be split across two channels — MCP tool calls and arbitrary Python — making it harder to replay, debug, or evaluate the verification workflow.

**4. Potential benefit: richer interactive analysis.** The one upside would be enabling exploratory analysis — e.g., the agent could write a cell that filters facts by stance, computes custom statistics, or runs a targeted PubMed query with custom logic, all visible in the notebook. This would make the notebook an *interactive lab notebook* rather than a *static report*.

**5. Iteration limit bypass risk.** The `check_sufficiency` tool enforces an 8-iteration cap inside `mcp_tools.py`. If the agent called the classifier directly from a notebook cell (`from pkevolve.verification.classifier import SufficiencyClassifier`), it would bypass this guard entirely.

In short: the current design deliberately keeps the notebook as a **view layer** to preserve state consistency, auditability, and iteration control. Merging the two would trade those guarantees for more flexible in-notebook exploration.