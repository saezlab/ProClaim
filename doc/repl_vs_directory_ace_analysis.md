# REPL vs Directory-of-Files: ACE-Informed Analysis

Compares our persistent-REPL evidence programming approach against the
directory-of-files paradigm used by coding agents, evaluated through the
lens of **ACE (Agentic Context Engineering)** (Zhang et al., ICLR 2026,
https://openreview.net/forum?id=eC4ygDs02R).

---

## Background: Two Paradigms

### Directory-of-files (coding-agent style)

The agent treats state as **files in a workspace directory** — markdown,
JSON, config — manipulated via OS-level read/write/search tools.  This is
the pattern used by coding agents (Claude Code, Cursor, Devin) and aligns
with ACE's structured playbook model.

### Persistent Python environment (our approach)

The agent executes arbitrary Python in a long-lived Jupyter kernel via
`nb_execute`.  State lives as Python objects (`EvidenceState`).  Code cells
form the audit trail.  See `implementation_plan.md` and `sdk_vs_repl_modes.md`.

---

## Trade-off Summary

| Dimension | Directory-of-files | Persistent REPL (ours) |
|---|---|---|
| **Crash resilience** | Every write is durable; no kernel to die | Kernel death loses all in-memory state; must replay or reload checkpoint |
| **Diffability** | `git diff` on markdown/JSON is natural | `.ipynb` diffs are opaque JSON blobs |
| **Hidden state** | None — all state is in files | Kernel variables are invisible to external tools/humans |
| **Parallelism** | Multiple agents can read/write with file locking | Single kernel = single execution thread |
| **Computation** | No in-process Python; must shell out or call services | Rich in-band computation (torch, scispaCy, SBERT, PubMed API) |
| **ML model persistence** | Models must reload per invocation (or require external service) | Singleton model cache (`model_registry.py`) survives across iterations |
| **Exploratory analysis** | Requires pre-built tools or scripts | Agent writes ad-hoc Python freely |
| **Tool surface** | Many fine-grained tools (read, write, search, list, append) | Minimal: `nb_execute` + render helpers |
| **Serialization cost** | JSON round-trips on every access for complex objects | Direct Python object access, zero serialization |
| **Resumability** | Trivial: re-read files | Must replay cells or reload from `evidence_state.json` |
| **ACE playbook alignment** | Direct: playbook = structured file, delta updates = file edits | Indirect: state is auto-saved JSON but not a structured playbook |

---

## ACE Alignment Assessment

### Where our approach aligns well

1. **Structured state persistence.** `EvidenceState` with auto-save to
   `evidence_state.json` acts as ACE's playbook equivalent.  Incremental
   mutations (`add_paper`, `add_fact`) avoid monolithic rewriting, preventing
   data-layer context collapse.

2. **Execution-level feedback.** ACE prioritizes execution traces over
   outcome-level scores.  Our system returns concrete outputs per
   `nb_execute` (paper counts, fact stances, sufficiency scores) that guide
   the next step — fine-grained execution feedback.

3. **Modular separation.** ACE's Generator/Reflector/Curator maps to our
   search/extract/sufficiency-check loop.  The Claude Agent SDK is the
   Generator, the MLP sufficiency classifier is a structured Reflector, and
   gap-query formulation acts as the Curator.

### Identified weaknesses (gaps vs ACE)

**W1. No evolving system prompt (playbook).**
ACE accumulates strategies and heuristics across tasks as itemized
bullet points.  Our `SYSTEM_PROMPT` in `evidence_programming.py` is
**static**.  If the agent discovers that "for kinase claims, searching
for 'phosphorylation substrate' works better than 'activates'", that
insight dies with the session.  ACE would persist it for future claims.

**W2. In-session context collapse vulnerability.**
After 8 iterations × (search + extract + features + sufficiency + gaps),
the agent's conversation history grows large.  The LLM must track kernel
variables, searched papers, and remaining gaps from conversation context
alone.  ACE addresses this with structured, itemized playbook entries that
are read from file rather than reconstructed from conversation memory.

**W3. No cross-session delta learning.**
ACE's core value is that strategies improve over multiple task episodes.
Our system starts from scratch for each claim.  There is no mechanism to
accumulate insights like "this PubMed query pattern yielded high-quality
papers for protein interaction claims" across runs.

**W4. Monolithic evidence state vs itemized playbook.**
`evidence_state.json` is one large blob.  ACE uses itemized bullets with
metadata (usage count, confidence, category).  With 50 papers and 200
facts, the agent cannot selectively retrieve "just the REFUTE facts for
subclaim 2" without writing Python to filter.  A directory-of-files
approach (`facts/refute/subclaim_2.md`) would make evidence directly
addressable.

**W5. Kernel fragility.**
If the Jupyter kernel dies (OOM from scispaCy, segfault, timeout),
all in-memory state is lost.  Recovery requires replaying cells or
reloading from checkpoint JSON.  A directory-based approach has no
analogous single point of failure.

**W6. Opaque audit trail.**
A new agent (or human reviewer) looking at the `.ipynb` sees code that
was run, not the current state.  They must mentally or actually replay
cells.  Directory-based state files are directly inspectable snapshots.

---

## Hybrid Recommendation

Our problem domain requires **heavy computation** (NLI, SBERT, scispaCy,
MLP classifier, PubMed API, full-text retrieval).  A pure directory-of-files
agent cannot do this without an external compute service.  The REPL is
justified for the computational layer.

The ACE-aligned improvement is to **layer a directory-of-files structure
on top of the REPL**, not replace it:

| Component | Current (REPL) | ACE-aligned hybrid |
|---|---|---|
| State persistence | `evidence_state.json` (adequate) | Same, or split into `papers/`, `facts/`, `gaps/` directory tree |
| Computation | `nb_execute` in kernel | Keep — required for ML models and API calls |
| Audit trail | Notebook cells (`.ipynb`) | Markdown files: `workspace/log/iteration_N.md` |
| Strategy accumulation | None (static prompt) | Persistent `playbook.md` updated across claim sessions via delta updates |
| Agent working memory | Conversation history | `workspace/status.md` — always-current file the agent reads to re-orient |

---

## References

- Zhang et al. "Agentic Context Engineering: Evolving Contexts for
  Self-Improving Language Models." ICLR 2026.
  https://openreview.net/forum?id=eC4ygDs02R
- `doc/REPL_issues.md` — notebook as passive view layer
- `doc/sdk_vs_repl_modes.md` — Mode A (SDK) vs Mode B (REPL)
- `doc/implementation_plan.md` — RLM REPL architecture
