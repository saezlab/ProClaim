# RLM Evidence Verification — Configuration, Context Analysis & Bottlenecks — 2026-03-04

**Branch:** `RLM`

## Summary

Three areas of work on the REPL-based evidence verification system: (1) centralised `pydantic-settings` configuration replacing scattered `argparse`/`os.getenv()`, (2) comprehensive context window utilisation analysis across orchestrator, subagent, and evidence-state layers, and (3) identification of full-text retrieval as the primary bottleneck limiting evidence quality. A `--subagent-model` CLI argument was added to decouple the inner `llm()` callable from the outer agent model. Analysis of seven context stress scenarios confirmed that orchestrator conversation growth and sparse-topic wasted turns are the most immediate risks, while the REPL architecture's external-memory design is validated as forward-looking for when full-text retrieval improves.

---

## New Files

| File | Purpose |
|------|---------|
| `src/pkevolve/verification/config.py` | Centralised `pydantic-settings` config: `APISettings`, `LLMSettings`, `VerificationSettings`. Factory methods: `from_cli()`, `from_yaml()`, `save_yaml()`, `build_sdk_env()`. Module-level `get_settings()` singleton. |
| `experiments/example_config.yaml` | Example YAML config for reproducible experiments. `claim` intentionally absent (CLI-only). |
| `doc/sdk_vs_repl_modes.md` | SDK (Mode A) vs REPL (Mode B) architecture comparison and academic reproducibility guidance. |
| `scripts/start_vllm_ihpc.sh` | One-command vLLM deployment on EBI HPC via SLURM (`--start`, `--stop`, `--reconnect`, `--status`, `--logs`). |
| `scripts/vllm_node_setup.sh` | Compute-node companion: GPU detection, port allocation, vLLM launch via Singularity. |

## Modified Files

| File | Changes |
|------|---------|
| `scripts/verification/demo_evidence_programming.py` | Replaced ~80 lines of manual `argparse` + `_build_env()` with `VerificationSettings.from_cli()`. Added `--subagent-model` CLI arg. `SYSTEM_PROMPT` uses `{subagent_model}` for inner `llm()`. |
| `src/pkevolve/verification/repl_orchestrator.py` | Added `cfg: VerificationSettings` and `subagent_model` params to `verify_claim_repl()`. Passes `subagent_model` to `inject_prelude(llm_model=...)`. |
| `src/pkevolve/verification/orchestrator.py` | Added `cfg` kwarg; replaced inline `_build_env()` with `cfg.build_sdk_env()`. |
| `src/pkevolve/verification/full_text.py` | `_get_unpaywall_email()` reads from `get_settings().api.unpaywall_email` with env-var fallback. |
| `src/pkevolve/verification/__init__.py` | Lazy imports and `__all__` entries for config classes. |
| `pyproject.toml` | Added `pydantic-settings>=2.0`, `pyyaml>=6.0`. Added `[project.optional-dependencies] fulltext`. |

---

## 1. Centralised Configuration (`config.py`)

### Architecture

```
CLI flags  ──►  ┌─────────────────────────────────────┐
                 │     VerificationSettings.from_cli()  │
                 │                                      │
YAML file  ──►  │  priority:                           │
(--config)       │    1. CLI flags                      │
                 │    2. Environment variables / .env    │
                 │    3. YAML config file                │
                 │    4. Field defaults                  │
.env file  ──►  │                                      │
                 └──────────────┬──────────────────────┘
                                │ cfg: VerificationSettings
                                │
          ┌─────────────────────┼──────────────────────────┐
          │                     │                           │
          ▼                     ▼                           ▼
   cfg.build_sdk_env()    cfg.model              cfg.save_yaml()
   → Mode A (SDK)         cfg.api_key            → reproducibility
                          cfg.openai_base_url       snapshot
                          → Mode B (REPL)
```

### Settings Hierarchy

```
VerificationSettings
├── api: APISettings
│   ├── glm_api_key        (from GLM_API_KEY env)
│   ├── zai_api_key        (from ZAI_API_KEY env)
│   ├── openai_api_key     (from OPENAI_API_KEY env)
│   ├── unpaywall_email
│   └── elsevier_api_key
├── llm: LLMSettings
│   ├── model              (outer agent model)
│   ├── subagent_model     (inner calls; defaults to model)
│   ├── openai_base_url / anthropic_base_url
│   └── temperature
├── claim                  (CLI-only, excluded from YAML)
├── mode                   (sdk | repl)
├── max_iterations, sufficiency_threshold
├── max_turns, max_output_chars
└── output_dir, notebook_path, verbose
```

### Key Design Decisions

- **pydantic-settings over Hydra/OmegaConf:** The project already uses Pydantic heavily (`data_models.py`). pydantic-settings adds env-var and `.env` merging natively without new concepts. Hydra's decorator-based approach would clash with the existing `argparse` CLI convention.
- **Three-model hierarchy (API / LLM / Verification):** Separates concerns — API keys are never mixed with model params, and both are isolated from workflow settings. Each sub-model can be tested independently.
- **`claim` excluded from YAML:** Claims vary per run (batch experiments iterate over many edges). `from_yaml()` warns and strips it; `save_yaml()` omits it; `from_cli()` makes `--claim` required.
- **Backward-compatible `cfg` kwarg:** `verify_claim_repl()` and `verify_claim()` accept an optional `cfg` parameter. Old callers passing explicit kwargs still work unchanged.
- **`build_sdk_env()` consolidation:** The 8-line SDK env dict (with `ANTHROPIC_AUTH_TOKEN`, beta disabling, timeout) was duplicated in 3 files with slight divergences. Now a single method on `VerificationSettings`.
- **`get_settings()` singleton for library code:** Functions like `_get_unpaywall_email()` in `full_text.py` need config but aren't passed a `cfg` object. The `@lru_cache` singleton reads from env vars on first access.
- **YAML via built-in pydantic-settings support:** pydantic-settings 2.12+ includes `YamlConfigSettingsSource`. PyYAML was already installed (transitive dependency).

---

## 2. Context Window Utilisation Analysis

### Architecture: Where Context Lives

```
┌─────────────────────────────────────────────────────────┐
│  ORCHESTRATOR CONTEXT (LLM window)                      │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ System prompt              ~1,250 tokens (fixed) │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ Turn 1: assistant code     ~200-800 tokens       │    │
│  │ Turn 1: user kernel output ~0-3,000 tokens       │    │
│  │ ...                                              │    │
│  │ Turn N: assistant code                           │    │
│  │ Turn N: user kernel output                       │ ←──── GROWS UNBOUNDEDLY
│  └─────────────────────────────────────────────────┘    │
│  Total after 30 turns: ~46K tokens                      │
└─────────────────────────────────────────────────────────┘
                        │
                        │ nb_execute / code blocks
                        ▼
┌─────────────────────────────────────────────────────────┐
│  JUPYTER KERNEL (external memory — NOT in LLM context)  │
│                                                         │
│  state.papers    : dict[str, PaperRecord]               │
│  state.facts     : list[Fact]                           │
│  state.coverage  : dict[str, float]                     │
│  state.synthesis : dict[str, str]                       │
│                                                         │
│  Measured: 3K–38K tokens across runs                    │
│  Bounded only by disk (auto-saved to JSON)              │
└─────────────────────────────────────────────────────────┘
                        │
                        │ extract_and_add_facts → llm(prompt)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  SUBAGENT CALLS (stateless, single-shot)                │
│                                                         │
│  Each call: fresh prompt → LLM → parse response         │
│  Paper text truncated to 16K chars (~4K tokens)         │
│  No history accumulation                                │
│  Max ~4,500 tokens per call                             │
└─────────────────────────────────────────────────────────┘
```

### Orchestrator (outer agent) — grows linearly, unbounded

| Component | Tokens | Notes |
|---|---|---|
| System prompt (REPL mode) | ~1,250 | Fixed; includes schema docs (~620 tokens) |
| System prompt (SDK mode) | ~1,750 | Larger due to setup code template |
| Per turn (assistant — code block) | 200–800 | Compact Python code |
| Per turn (user — kernel output) | 0–3,000 | Capped at `MAX_OUTPUT_CHARS=12,000` chars |
| **Worst case, 30 turns** | **~46K** | Approaches limit for smaller models |

The `messages[]` list in `verify_claim_repl()` is **append-only with no summarisation or eviction**.

### Subagents (inner `llm()`) — safely bounded

| Subagent | Tokens per call | Bounded by |
|---|---|---|
| `extract_facts` | ~4,000–4,500 | `paper_text[:16000]` hard truncation |
| `synthesize_subclaim` | ~400–2,000 | Number of facts (typically <20) |
| `detect_conflicts` | ~400–4,000 | Number of facts |
| `formulate_gap_queries` | ~400–1,000 | Number of gaps |

Subagents are stateless single-shot calls — they construct a fresh prompt each time and do not accumulate context. **No memory management needed.**

### Evidence State (kernel memory) — measured from real runs

| Run | Papers | Facts | Evidence tokens | State JSON |
|---|---|---|---|---|
| glm-5 (7 iterations, 69 papers) | 69 | 13 | ~38K | 186 KB |
| local_state (2 iterations, 24 papers) | 24 | 15 | ~23K | 107 KB |
| glm-4.6 (1 iteration, 12 papers) | 12 | 7 | ~21K | 90 KB |
| latest (0 iterations, 0 facts — 429 issue) | 12 | 0 | ~3K | 18 KB |

All runs use **<30% of a 128K window** and **<20% of a 200K window**.

### Compression Gap

`SufficiencyPreservingCompressor` is L1 only (exact `(text, stance)` deduplication). In practice this rarely fires because LLMs almost never generate identical fact strings. The system prompt mentions a 40K token budget but the compression cannot meaningfully reduce state when hit.

---

## 3. Full-Text Retrieval — The Actual Bottleneck

### Retrieval Success Rates

| Run | Papers | Full text retrieved | Abstract only | Success rate |
|---|---|---|---|---|
| glm-4.6 (1 iter) | 12 | 5 | 7 | 41% |
| local_state (2 iter) | 24 | 5 | 19 | 20% |
| glm-5 (7 iter) | 69 | 6 | 63 | **8%** |
| latest (0 iter) | 12 | 0 | 12 | 0% |

The 3-layer fallback chain (PMC Open Access → INDRA → Unpaywall+PDF) fails for the majority of papers. Success rate drops as later iterations search for more niche papers.

### Text Actually Used vs Available Capacity

| Run | Text used by subagent | Total evidence tokens | 128K capacity used | 200K capacity used |
|---|---|---|---|---|
| glm-4.6 | ~20K tokens | ~21K | **16%** | **10%** |
| glm-5 | ~36K tokens | ~38K | **29%** | **19%** |
| local_state | ~18K tokens | ~24K | **18%** | **12%** |

### The 16K Truncation: Minimal Waste in Practice

| Run | Full texts >16K chars | Chars wasted by truncation |
|---|---|---|
| glm-4.6 | 2/5 | 1,700 chars (~425 tokens) |
| glm-5 | 2/6 | 1,700 chars (~425 tokens) |
| local_state | 3/5 | 12,345 chars (~3K tokens) |

The truncation itself wastes very little because most full texts are already under 16K chars. The real issue is that **full-text retrieval fails for 60–90% of papers**, so most papers contribute only ~1K chars of abstract rather than ~15K chars of full text.

**What-if: all papers at full text** — With 69 papers all at ~15K chars, the total would be ~259K tokens (exceeds 200K). But the realistic mix of abstract + available full text is ~38K tokens, which fits comfortably in 128K.

### Two Distinct Problems

**Problem 1: Low full-text retrieval rate (the bigger issue).** The 3-layer chain fails because PMC covers only Open Access papers (~40% of biomedical literature), INDRA requires an Elsevier API key with limited publisher coverage, and Unpaywall requires DOIs (not always stored in `PaperRecord`).

**Problem 2: Subagent text truncation (conservative but low-impact).** The hard truncation at `paper_text[:16000]` in `subagents.py` was set for 8K–32K models. For 128K+ models, raising to 50K chars (~12K tokens) is safe. However, only 2–3 papers per run exceed 16K, making this a low-impact change until retrieval improves.

### Retrieval Improvements (highest ROI)

- Store DOIs in `PaperRecord` during PubMed search (needed for Unpaywall)
- Add Europe PMC as Layer 1b (broader OA XML coverage)
- Add Semantic Scholar API as Layer 2b (free PDF links)
- Add bioRxiv/medRxiv preprint fetching
- Consider institutional proxy support for paywalled papers

---

## 4. Context Stress Scenarios

### Summary

| Scenario | Primary stress point | Likely today? |
|---|---|---|
| Broad full-text retrieval (60%+) | Subagent cost (~900K tokens aggregate) | No |
| Multi-mechanism claims (4–6 subclaims) | Orchestrator turns (24–36 turns, 40–60K tokens) | **Yes** |
| Contradicted claims | Evidence state + orchestrator reasoning | **Yes** |
| Sparse/emerging topics | Wasted turns on failed searches | **Yes** |
| Batch verification (50+ claims) | No cross-claim evidence sharing | Not yet supported |
| Agent reasoning chains (Mode A) | Invisible context from chain-of-thought | **Yes** |
| Tables/supplementary data | Low information density per token | Partially |

### Scenario Details

**Multi-mechanism claims** — Claims involving indirect regulation (e.g., "EGFR activates STAT3 via JAK2 phosphorylation leading to BCL2 transcription") require 4–6 subclaims with independent literature searches, pushing to 24–36 turns and 40–60K tokens of orchestrator context.

**Contradicted claims** — When SUPPORT and REFUTE facts are in roughly equal numbers, conflict detection produces many pairs, sufficiency gaps trigger additional search rounds, and longer synthesis paragraphs push evidence state and orchestrator reasoning tokens higher.

**Sparse/emerging topics** — Few PubMed results lead to many turns wasted on unsuccessful searches. 20 turns of failed searches × ~1K tokens/turn = 20K tokens of noise in the orchestrator conversation.

**Agent reasoning chains (Mode A)** — Chain-of-thought tokens count against context in Claude Agent SDK mode. Cumulative input cost across 30 turns (full history resent each turn): approximately $\sum_{i=1}^{30}(1250 + 1500i) \approx 735\text{K}$ total input tokens billed.

**Batch verification** — No mechanism to share evidence across claims. 50 related claims would perform 50 independent PubMed searches with massive overlap.

### Implications for Architecture

1. **The REPL architecture's value is forward-looking** — it doesn't save context today (evidence fits in 128K), but it will when full-text retrieval improves and the system handles 100+ papers with actual full text
2. **Orchestrator conversation management is the urgent fix** — sliding window + state summary handles multi-mechanism and sparse-topic scenarios
3. **Section-aware extraction from papers would improve token efficiency** — extracting only Introduction/Results/Discussion would cut per-paper tokens by 40–60%
4. **The subagent 16K truncation is a non-issue for current models** — raise it to match `DEFAULT_MAX_CHARS` (50K) and let the model see everything retrieved
5. **Batch claim verification needs a shared evidence cache** — the biggest context waste at scale is cross-claim redundancy

---

## 5. Recommended Fixes (priority order)

### HIGH: Sliding window for orchestrator `messages[]`

The most immediate risk. After N turns, summarise the oldest K turns into a single progress message and drop the originals:

```python
# In repl_orchestrator.py, before each LLM call:
if len(messages) > WINDOW_THRESHOLD:  # e.g. 20 messages
    summary = get_evidence_summary(state)  # already exists, FREE
    messages = [
        messages[0],  # system prompt
        {"role": "user", "content": f"Progress so far:\n{summary}"},
    ] + messages[-KEEP_RECENT:]  # keep last 6-10 messages
```

Caps orchestrator context at ~system + summary + 5 turns ≈ 8–12K tokens regardless of run length.

### MEDIUM: Raise subagent truncation to 50K

Change `paper_text[:16000]` → `paper_text[:50000]` in `subagents.py`, matching `full_text.py`'s `DEFAULT_MAX_CHARS`. Trivial for 128K+ models. Low impact today (few papers exceed 16K) but correct for when retrieval improves.

### MEDIUM: Budget-aware `get_context(budget_tokens)`

Add priority-based truncation to `EvidenceState.get_context()`: facts first (small, high-value), then summaries, then abstracts, stopping at budget.

### MEDIUM: Semantic deduplication (L2 compression)

Embed facts with a small model (e.g., `all-MiniLM-L6-v2`), cluster by cosine similarity >0.9, keep one representative per cluster. Useful once runs consistently extract 30+ facts.

### LOW: Section-aware paper extraction

Extract only Introduction/Results/Discussion from full text, skip Methods/References/Supplementary. Would cut per-paper tokens by 40–60%.

### LOW: Paper summarisation after fact extraction

Replace stored full text with a short claim-relevant summary post-extraction. Would reduce the glm-5 run's evidence from ~38K to ~5K tokens.

### LOW: Full hierarchical L0/L1/L2 memory (future architecture)

For multi-claim batch verification or claims requiring 50+ papers:
- **L0 (in-context)**: System prompt + last 3–5 turns + state summary (~8K, fixed)
- **L1 (kernel memory)**: Full EvidenceState — all papers, facts, coverage
- **L2 (disk)**: Full text PDFs, trace logs, conversation history

The orchestrator sees only L0. It writes code to access L1/L2 as needed. This is already the implicit architecture — the fix is formalising the L0 boundary with explicit eviction.

---

## Bug Fixes

- **Subagent model mismatch:** The inner `llm()` callable in the kernel prelude was hardcoded to use `{model}` (the outer agent model, e.g., `glm-5`), but the local vLLM endpoint serves `openai/gpt-oss-120b`. Fixed by introducing `{subagent_model}` as a separate template variable and adding `--subagent-model` CLI argument that defaults to `--model` for backward compatibility.
- **Duplicated `_build_env()` in 3 files:** `demo_evidence_programming.py`, `orchestrator.py`, and `repl_orchestrator.py` each had their own copy of the SDK environment construction logic with slight divergences. Consolidated into `VerificationSettings.build_sdk_env()`.
