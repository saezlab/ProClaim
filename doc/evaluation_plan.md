# Evaluation Implementation Plan: Evidence Programming for Scientific Claim Verification

## 1. Evaluation Design Overview

The evaluation is split into two groups that test fundamentally different capabilities:

**Group 2 — Open retrieval + claim-level verdict (primary benchmarks)**
Systems search freely across PubMed, Semantic Scholar, or any available API. Evaluation is at the claim level against curated gold labels. Tests: the full evidence programming paradigm — iterative construction, gap-directed retrieval, sufficiency-aware stopping in an unconstrained search space. SIGNOR\* and connectomeDB\* are the **primary novel contribution** of this benchmark: no existing dataset provides claim-level labels for open-retrieval scientific verification.

**Group 1 — Constrained retrieval + verdict (secondary benchmarks)**
All systems retrieve from the same fixed corpus. Evaluation is at the (claim, evidence) pair level. Results are directly comparable to published numbers. Included as a secondary diagnostic: by holding retrieval fixed, failures can be attributed to reasoning rather than retrieval, establishing an upper bound on what a perfect retriever could achieve. SciFact-Open and CIViC-Fact are included for cross-system comparability and task decomposition, not as the primary evaluation target.

**Relationship between groups:** Group 2 is the primary evaluation — it tests whether a system can find and synthesise evidence autonomously. Group 1 is a secondary diagnostic that tells us whether failures in Group 2 stem from poor reasoning or poor retrieval.

---

## 2. Datasets

### Group 2: Open Retrieval (Claim-Level) — Primary

| Dataset | Retrieval Scope | Size | Label Schema | Eval Protocol |
|---------|----------------|------|--------------|---------------|
| **connectomeDB*** | Open (PubMed, Semantic Scholar, any API) | ~450 claims (sampled) | SUPPORTED / REFUTED | Predict claim-level verdict |
| **SIGNOR*** | Open | 67 edges (34 SUPPORTED, 24 REFUTED, 9 UNCERTAIN) | SUPPORTED / REFUTED / UNCERTAIN | Predict claim-level verdict |
| Matter-of-Fact (hard subset) | Open (Semantic Scholar) | ~1–2k claims (filtered from 4.4k) | FEASIBLE / INFEASIBLE → SUPPORT / REFUTE | Predict claim-level verdict |

- No corpus constraint — systems search freely.
- **Primary metric: Claim-level F1 against curated gold labels.**
- connectomeDB\* is the primary benchmark (largest, cleanest labels, ligand-receptor cell communication domain).
- SIGNOR\* is a secondary benchmark within this group (smaller, different domain: intracellular signaling regulation).
- Matter-of-Fact is a supplementary benchmark for domain breadth (materials science) and scale; hard-claim filtering applied to remove trivially verifiable cases.

### Group 1: Constrained Retrieval — Secondary

| Dataset | Corpus | Size | Label Schema | Eval Protocol |
|---------|--------|------|--------------|---------------|
| **SciFact-Open** | 500K S2ORC abstracts | 279 claims | SUPPORT / CONTRADICT | Retrieve abstract + predict verdict per (claim, abstract) pair |
| **CIViC-Fact** | 554 annotated publications | 1,434 claim-evidence pairs (1,119 distinct claims) | SUPPORTS / DOES NOT SUPPORT / NEI / CONFLICTING | Retrieve paper + predict verdict per (claim, paper) pair |

- All systems search the same corpus — this controls for retrieval source and isolates architectural differences.
- Secondary metric: Label F1 (and optionally Label+Retrieval F1, requiring both correct retrieval and correct verdict).
- SciFact-Open is included for comparability with prior work (FIRE, SAFE, etc.) and to diagnose reasoning failures independently of retrieval.
- CIViC-Fact is included for scale, statistical power, and domain breadth (precision oncology).

**Rationale for including Group 1 datasets as secondary:**

SciFact-Open and CIViC-Fact are included as secondary diagnostics, not as the primary evaluation target. The primary contribution of this benchmark is Group 2 (connectomeDB\*, SIGNOR\*), which is the first claim-level evaluation for open-retrieval scientific verification in these domains.

1. **Task decomposition / verification upper bound.** Open-retrieval verification = retrieval + reasoning. Group 1 isolates the reasoning step with gold evidence, letting us separately diagnose whether Group 2 failures come from retrieval errors or reasoning errors. Without this decomposition, it is impossible to determine whether a system that fails in open retrieval does so because it cannot find the evidence or because it cannot reason over it.

2. **Community comparability.** SciFact-Open is the established NLP benchmark for scientific claim verification. Reporting results on it anchors the evaluation in the prior literature and enables direct comparison with FIRE, SAFE, Factcheck-GPT, and OpenScholar — making the paper's contribution visible to the NLP community.

3. **Domain breadth.** Together, the four datasets span general biomedical science (SciFact), precision oncology (CIViC), intracellular signaling regulation (SIGNOR), and ligand-receptor cell communication (ConnectomeDB).

### connectomeDB\* Construction

Source: connectomeDB2025 (Liu et al., Nucleic Acids Research 2026, doi:10.1093/nar/gkaf1108). Contains 3,579 verified ligand-receptor pairs and 2,900+ rejected pairs from other databases.

Three claim categories:
1. **SUPPORTED** — sampled from 3,579 verified pairs. Stratify by difficulty: pairs with many supporting articles (easy) vs. single recent publication (hard).
2. **REFUTED** — sampled from 2,900+ rejected pairs that lacked primary experimental evidence.
3. **Hard negatives** — plausible but false pairs generated by swapping ligands/receptors within the same gene family.

Target: ~200 SUPPORTED, ~200 REFUTED, ~50 hard negatives. Describe the sampling procedure in the paper.

### SIGNOR\* Construction

Source: SIGNOR database (LoSurdo et al., 2025). Ground-truth incorrect edges identified by comparing SIGNOR releases 2018-2025. Correct edges filtered from October 2025 release, ranked by reference counts, deduplicated, and filtered to remove easy cases. Expert adjudicated. 67 total edges.

### Matter-of-Fact (Supplementary — Open Retrieval)

**Source:** Jansen et al., EMNLP 2025 (arXiv:2506.04410). 4,446 test claims extracted from arXiv materials science papers (superconductors, semiconductors, batteries, aerospace materials), published January 2024–April 2025. Balanced 50/50 feasible/infeasible. Labels validated by domain expert (κ = 0.86; 99% agreement after resolution). CC-BY 4.0. Repository: https://github.com/cognitiveailab/matter-of-fact

**Label mapping:** feasible → `SUPPORT`; infeasible → `REFUTE`. (Two-class only; no NEI.)

**Label construction:** Positive claims extracted directly from full-text results reported in the source paper. Negative claims are LLM-generated matched counterparts instructed to be scientifically infeasible with neutral framing. Both are confirmed verifiable against the literature: oracle RAG (Sonnet 3.7) achieves 100% accuracy on the claim verification task (temporally unrestricted), and domain expert agrees with LLM-generated labels for 99% of claims after resolution.

**Evaluation mode:** Use as a standard claim-level open-retrieval benchmark without temporal filtering (the temporally-unrestricted oracle mode). This avoids the per-claim date-cutoff complexity of their feasibility assessment task and is appropriate because Evidence Programming retrieves evidence to verify claims rather than predict future experimental results.

**Subset construction — filtering to hard claims only:**

The full 4,446-claim test set includes many claims that are trivially verifiable by any model (best published baseline = 72% accuracy on the feasibility task; oracle claim verification ≈ 100%). To focus evaluation on claims that genuinely require deep retrieval and reasoning:

- **Primary approach:** Authors contacted (March 2026) to request per-claim baseline predictions from the paper. If received, exclude claims correctly classified by all published baselines (gpt-4o-mini CoT, o4-mini CoT, Claude Sonnet 3.7 CoT) and evaluate only on the remaining hard subset.
- **Fallback if no response:** Run 2–3 small open-weight LLMs (e.g., Llama-3.1-8B, Mistral-7B) at temperature=0 on all 4,446 test claims with a minimal prompt (no retrieval). Remove claims unanimously and consistently correct across all SLMs across 3 runs. The surviving hard subset — claims that SLMs cannot resolve from parametric knowledge alone — becomes the evaluation target.

Target hard subset: ~1,000–2,000 claims (estimated based on published ~50–55% SLM accuracy → ~45–50% of claims are SLM-errors, keeping only the intersection of SLM failures).

**Rationale for inclusion:** Adds domain breadth (materials science) and scale (~1k hard claims vs. ~550 total biology claims) to the open-retrieval evaluation. Confirms generalization of Evidence Programming beyond biomedical domains. The systematic label construction and strong expert validation make it a credible addition despite the synthetic negative construction method.

---

## 3. Compared Systems

### Baseline Table Structure

```
No retrieval:           Random, LLM-only
Static retrieval:       LLM + 5 search results, Fixed-5 RAG (BM25), Fixed-10 RAG (BM25)
Prompting strategy:     ReAct
Specialised RAG:        OpenScholar
LLM verification pipes: Factcheck-GPT, SAFE, FIRE
Ours:                   Evidence Programming
```

### 3.1 Random

**What it tests:** Chance-level floor.

**Implementation:**
- Randomly assign SUPPORT / REFUTE / NEI to each claim.
- Cost: 0 tokens, 0 LLM calls.
- Expected: ~33% F1 for balanced 3-class.
- Implementation time: 30 minutes.

### 3.2 LLM-only (No Retrieval)

**What it tests:** Parametric knowledge ceiling. Measures how much the LLM knows from pretraining and quantifies prior-dominated failure rate. This is the only baseline run across **multiple models** — all other baselines use a single backbone LLM to isolate architectural differences.

**Models:**

| Model | Parameters | Access | Rationale |
|-------|-----------|--------|----------|
| **Gemini-3** | — | Google API (`google-genai` SDK) | Frontier proprietary; strong biomedical pretraining |
| **Claude Sonnet 4.6** | — | Anthropic API | Same backbone as Evidence Programming — cleanest architecture comparison |
| **GPT-OSS-120B** | 120B | OpenAI-compatible endpoint | Large open-source; already integrated in codebase |
| *Qwen3-32B (optional)* | 32B | Self-hosted vLLM (2× RTX A6000, TP=2) | Open-weight size-scaling comparison; requires local GPU |

**Implementation:**
- Single LLM call per claim with no retrieved context.
- Prompt asks the LLM to classify based solely on its knowledge.
- Use temperature=0 for deterministic output.
- Run each of the models above on every dataset; report per-model results.
- Cost: 1 LLM call per claim, ~500–1000 tokens per claim per model.
- Implementation time: 1 hour per model.

### 3.3 LLM + 5 Search Results

**What it tests:** Minimal augmentation — simulates pasting search snippets into an LLM. Zero-engineering baseline.

**Implementation:**
- Use the raw claim as a PubMed API query.
- Retrieve top 5 results as search snippets (title + snippet, NOT full abstracts).
- Single LLM call with snippets concatenated into context.
- Key distinction from Fixed-k RAG: uses raw claim as query (no BM25 over indexed corpus) and uses search snippets (not full abstracts).
- Cost: 1 API search + 1 LLM call per claim.
- Implementation time: 1-2 hours.

### 3.4 Fixed-5 RAG (BM25) and Fixed-10 RAG (BM25)

**What it tests:** Standard retrieve-then-verify pipeline with fixed budget. Primary lower bound for adaptive approaches. Running at k=5 and k=10 shows diminishing returns.

**Implementation:**
- Pre-index each dataset's corpus with BM25 (rank_bm25 library).
- For each claim: BM25 retrieval → concatenate top-k full abstracts → single LLM call for verdict.
- Use the SAME verification prompt as other baselines (only the evidence context varies).
- Corpus indexing per dataset:
  - SciFact-Open: 500K S2ORC abstracts
  - CIViC-Fact: 554 publications
  - SIGNOR\*/connectomeDB\*: PubMed abstracts linked to relevant interactions
- Cost: 1 LLM call per claim. BM25 is local (~0 cost). Token cost scales with k.
- Implementation time: 2-3 hours (including corpus indexing).

### 3.5 ReAct (ICLR 2023)

**What it tests:** Whether structured evidence programming adds value over unconstrained agentic reasoning with the same tools. The "why not just use a ReAct agent?" baseline.

**Why ReAct over alternatives (IRCoT, Self-RAG):**
- ReAct gives the agent maximum freedom — it can reason, search, read in any order. This makes it the fairest architectural comparison: if Evidence Programming beats ReAct, the improvement comes from structure (evidence state, sufficiency classifier), not from constraining the agent.
- Works with any backbone LLM via prompting — no special model or fine-tuning required.
- IRCoT forces retrieval at every CoT step (a structural constraint that may be suboptimal for claim verification). Self-RAG requires fine-tuning custom reflection tokens into the model, making fair comparison infeasible.

**Implementation:**
- Purely a prompting strategy: Thought → Action → Observation loop.
- Agent has access to same tools as Evidence Programming: `search_pubmed(query)`, `read_abstract(pmid)`.
- Agent decides on its own when to stop — NO external sufficiency signal.
- Use native tool-use API (Claude/GPT-4o tool calling) rather than parsing text.
- Set `max_steps=10` (tune so median token usage is comparable to Evidence Programming).
- Fresh conversation per claim (no memory between claims).

**Key tracking:** How many search queries does ReAct issue? How many papers does it read? Does it stop too early or too late?

**What ReAct lacks vs. Evidence Programming:**

| Dimension | ReAct | Evidence Programming |
|-----------|-------|---------------------|
| What guides next action? | LLM's own reasoning | Sufficiency classifier gap predictions |
| When does it stop? | LLM decides ("I have enough") | Classifier confidence ≥ threshold |
| Evidence organisation | Flat conversation history | Structured state (facts, subclaims, coverage) |
| Cross-paper synthesis | Implicit in LLM context | Explicit per-subclaim synthesis |
| Memory across steps | Conversation grows (context fills up) | Evidence state persists on disk |

- Cost: Variable (5-15 LLM calls per claim typical).
- Implementation time: 3-4 hours.

### 3.6 OpenScholar (Nature 2025)

**What it tests:** Whether a system with a better retriever (trained on 45M scientific papers, domain-specific reranking, self-feedback) beats your system's better architecture (sufficiency-guided iteration) despite having a weaker retriever.

**Source:** Fully open-source. GitHub: https://github.com/AkariAsai/OpenScholar

**Components:**
1. Datastore: 45M open-access papers from Semantic Scholar with ~250M passage embeddings
2. Trained retrievers and rerankers for scientific literature
3. OpenScholar-8B: Llama 3.1 8B fine-tuned for scientific synthesis
4. Iterative self-feedback generation loop

**Implementation options:**
- **Option A (recommended):** Use OpenScholar-8B with their full pipeline (retrieval → reranking → generation → self-feedback).
- **Option B:** Use OpenScholar pipeline with GPT-4o backbone (OS-GPT4o) for closer model-family comparison.

**Task adaptation:** OpenScholar is designed for literature synthesis questions, NOT claim verification. Must frame claims as questions ("What does the scientific evidence say about [claim]? Is it supported or refuted?") and parse responses into verdict format. Acknowledge this framing mismatch in the paper — it strengthens the argument if a task-specific system outperforms a general-purpose tool.

**Setup requirements:**
1. Clone repository
2. Download OpenScholar-DataStore (passage embeddings)
3. Download reranker model
4. GPU required for 8B model inference (or use API for OS-GPT4o variant)

**Model note:** Different model family (Llama 3.1 8B or GPT-4o) from your backbone. Acknowledge in paper. Report as "OpenScholar-8B" or "OS-GPT4o".

- Cost: ~5-10 LLM-equivalent calls per claim (self-feedback loop typically 2-3 iterations).
- Implementation time: 6-8 hours (mainly environment setup and datastore download).

### 3.7 Factcheck-GPT (Findings of EMNLP 2024)

**What it tests:** Whether iteration matters at all. Factcheck-GPT is a well-engineered static pipeline — decompose → decontextualise → check — with no feedback loop. If your system beats Factcheck-GPT, adaptation matters.

**Implementation:**
Three-stage static pipeline:
1. **Decompose:** LLM splits claim into atomic checkworthy statements.
2. **Decontextualise:** LLM rewrites each statement to be self-contained.
3. **Check:** For each statement, retrieve evidence (Google Search in original; adapt to PubMed) and classify in a single pass.

No iteration between stages. No feedback from the checking stage back to retrieval. Each statement checked independently (like SAFE but single-pass).

- Cost: ~3 LLM calls per atomic statement (decompose + decontextualise + check). For a claim with 3 subclaims: ~9 LLM calls total.
- Implementation time: 3-4 hours.

### 3.8 SAFE (NeurIPS 2024)

**What it tests:** Per-fact independent iterative verification vs. your shared evidence state across subclaims. SAFE decomposes claims into atomic facts, then each fact gets its own multi-round search loop — but facts are verified independently with no shared evidence.

**Implementation:**
Three stages:
1. **Decompose** claim into atomic facts (1 LLM call).
2. **Per-fact iterative search loop** (up to 3 rounds per fact): generate query → search PubMed → reason over results → classify or search again.
3. **Aggregate** per-fact verdicts into claim-level verdict (majority vote weighted by confidence).

**Adaptation from original:**
- Original uses Google Search via Serper API → adapt to PubMed API.
- Original targets long-form text factuality → adapt prompt to scientific claim verification.
- Preserve core property: each atomic fact verified independently, no shared evidence state.

**Critical comparison point:** If paper X is relevant to both subclaim 1 and subclaim 3, SAFE might retrieve it twice (or miss it for subclaim 3). Evidence Programming's shared evidence state avoids this redundancy.

- Cost: (N_atomic_facts × max_rounds × ~3) LLM calls per claim. For 3 subclaims, 2 rounds each: ~18 LLM calls.
- Implementation time: 4-5 hours.

### 3.9 FIRE (Findings of NAACL 2025)

**What it tests:** Your learned MLP sufficiency classifier vs. LLM-internal confidence gating for stopping. The most direct competitor on the stopping criterion.

**Implementation:**
Core mechanism: LLM attempts verification → self-assesses confidence → if confident (≥ threshold), emit verdict; if uncertain, generate targeted query, retrieve, incorporate results, repeat.

Key components:
1. **Confidence self-assessment:** LLM outputs a confidence score alongside its verdict at each iteration.
2. **Query generation:** When uncertain, LLM generates a retrieval query targeting the identified uncertainty.
3. **Repetitive query prevention:** Fuzzy deduplication of queries to prevent loops.
4. **Stopping:** When self-assessed confidence ≥ threshold OR max iterations reached.

**Adaptation:**
- Original uses Google Search → adapt to PubMed API.
- Match `confidence_threshold=0.80` to your system's sufficiency threshold.

**Key tracking:** Compare distribution of stopping iterations between FIRE and Evidence Programming. If FIRE stops earlier with lower accuracy, the LLM's self-assessed confidence is miscalibrated — validating the need for an external learned signal.

- Cost: 2-3 LLM calls per iteration × 1-5 iterations = ~5-15 LLM calls per claim.
- Implementation time: 4-5 hours.

### 3.10 Evidence Programming (Ours)

As specified in the framework implementation plan. Structured evidence state + sufficiency classifier + gap-directed retrieval + iterative evidence construction via code generation.

---

## 4. Shared Infrastructure

Build these components first — every system depends on them.

### 4.1 Retrieval Backend

Unified retrieval module used by all baselines:
- `bm25_search(query, corpus, k)` — BM25 over pre-indexed corpus (Group 1).
- `pubmed_api_search(query, k)` — Live PubMed API (Group 2, agentic systems).
- `semantic_scholar_search(query, k)` — Semantic Scholar API (OpenScholar, Group 2).
- `get_abstract(pmid)` — Fetch full abstract by PMID.

### 4.2 LLM Backend

Same backbone LLM across all non-specialised baselines. Tracks cost automatically:
- Call count, total tokens (input + output), wall-clock time per claim.
- Use temperature=0 for deterministic output where applicable.
- Non-specialised baselines: LLM+search, Fixed-k RAG, ReAct, Factcheck-GPT, SAFE, FIRE, Evidence Programming.
- Specialised models (different family): OpenScholar-8B (Llama 3.1 8B) — report model size difference.

#### LLM-only Model Choices

The LLM-only baseline (§3.2) is the only baseline run across **multiple models**, since its purpose is to measure the parametric knowledge ceiling per model family.

| Model | Parameters | Access | Rationale |
|-------|-----------|--------|----------|
| **Gemini-3** | — | Google API (`google-genai` SDK) | Frontier proprietary; strong biomedical pretraining coverage |
| **Claude Sonnet 4.6** | — | Anthropic API | Matches Evidence Programming backbone — cleanest architecture comparison |
| **GPT-OSS-120B** | 120B | OpenAI-compatible endpoint | Large open-source model; already integrated in codebase |
| *Qwen3-32B (optional)* | 32B | Self-hosted vLLM (2× RTX A6000, TP=2) | Open-weight size-scaling comparison; requires local GPU |

**Authentication:**
- Gemini: `GEMINI_API_KEY` env var (falls back to `GOOGLE_API_KEY`).
- Claude: Anthropic API key.
- GPT-OSS-120B: `OPENAI_API_KEY`.
- Qwen3-32B: `api_key="EMPTY"`, `base_url="http://localhost:8000/v1"` (local vLLM).

**Implementation note:** `LLMBackend` (`experiments/baselines/shared/llm.py`) auto-resolves the correct API endpoint from the model name prefix via `_resolve_base_url()`. The `--base-url` CLI flag is only needed to override this. `response_format=json_object` is automatically skipped for `claude-*` models, which do not support it via the OpenAI-compatible proxy.

### 4.3 Evaluation Harness

Runs any baseline on any dataset, collects metrics:
- For each claim: run baseline, collect verdict + cost metrics.
- Reports mean ± std over 3 runs.
- Handles both Group 1 (pair-level) and Group 2 (claim-level) evaluation protocols.

### 4.4 Verdict Schema

All baselines output the same structured format:
```
Verdict:
  label: SUPPORT | REFUTE | NEI
  confidence: float 0-1
  evidence: list of PMIDs or text snippets
  reasoning: free-text explanation
```

### 4.5 Shared Verification Prompt

Use the SAME verdict-extraction prompt across Fixed-k RAG, LLM+Search, and the final verdict step of agentic systems. The only variable across baselines should be what evidence is provided and how it was gathered.

---

## 5. Metrics

### Primary Metrics (reported for all systems, all datasets)

| Metric | What It Measures |
|--------|-----------------|
| **Label F1** | Verification accuracy (macro F1 over label classes) |
| **Cost (tokens/claim)** | Total tokens (input + output) across all LLM calls |
| **LLM calls/claim** | Number of LLM API calls (orchestrator + subagents + all intermediate calls) |

### Secondary Metrics

| Metric | What It Measures |
|--------|-----------------|
| Latency (sec/claim) | Wall-clock time |
| Sufficiency rounds | Iterations before stopping (for adaptive systems) |
| Retrieval precision | Fraction of retrieved papers that are relevant (Group 1 only) |

### Cost Accounting Rules

- Count ALL LLM calls: orchestrator turns, subagent calls, query generation, relevance judgments, confidence assessment — not just the final verdict call.
- The sufficiency classifier (MLP) is NOT an LLM call — report separately.
- For OpenScholar: report inference cost for the 8B model or API cost for GPT-4o variant.

---

## 6. Fairness Checklist

Before running experiments, verify:

- [ ] Same LLM backbone for all non-specialised baselines
- [ ] Same verification prompt for verdict extraction across all systems
- [ ] Same corpus for BM25 retrieval in Group 1 (all systems search the same indexed corpus)
- [ ] Same output schema (Verdict) parsed the same way
- [ ] Same evaluation harness computing F1 identically
- [ ] Same max compute budget — no system gets dramatically more tokens without this being visible in the Cost column
- [ ] 3 runs per claim for variance estimation
- [ ] Cost tracking captures ALL LLM calls including intermediate steps
- [ ] OpenScholar model size difference acknowledged in paper
- [ ] Group 1 vs. Group 2 evaluation protocol clearly distinguished

---

## 7. Implementation Timeline

### Week 1: Foundation
| Task | Effort | Dependencies |
|------|--------|-------------|
| Shared infrastructure (retrieval backend, LLM backend, eval harness, verdict schema) | 2-3 days | None |
| Corpus indexing for all Group 1 datasets (BM25 over SciFact-Open 500K, CIViC-Fact 554 papers) | 1 day | Retrieval backend |
| connectomeDB\* claim sampling and construction | 1 day | connectomeDB2025 data download |
| Random + LLM-only baselines | 0.5 day | Shared infra |

### Week 2: Static + Prompting Baselines
| Task | Effort | Dependencies |
|------|--------|-------------|
| LLM + 5 search results | 0.5 day | Shared infra |
| Fixed-5 RAG + Fixed-10 RAG | 1 day | Corpus indexing |
| ReAct | 1-2 days | Shared infra |
| Factcheck-GPT | 1-2 days | Shared infra |

### Week 3: Agentic Baselines
| Task | Effort | Dependencies |
|------|--------|-------------|
| SAFE | 2 days | Shared infra |
| FIRE | 2 days | Shared infra |
| OpenScholar setup (download datastore, models, configure environment) | 2-3 days | GPU access |

### Week 4: Full Evaluation Runs
| Task | Effort | Dependencies |
|------|--------|-------------|
| Group 2 runs (primary): all systems on connectomeDB\* + SIGNOR\* (3 runs each) | 2-3 days | All baselines ready |
| Group 1 runs (secondary): all systems on SciFact-Open + CIViC-Fact (3 runs each) | 2-3 days | All baselines ready |
| Results aggregation, tables, cost-accuracy Pareto curves | 1-2 days | All runs complete |

**Total: ~4 weeks** from start to complete results tables.

---

## 8. Expected Results Narrative

The results should tell a progressive story, led by the primary open-retrieval evaluation.

**Group 2 (open retrieval, claim-level) — primary narrative:**
- This is the primary evaluation. Advantage of adaptive systems should grow in an unconstrained search space — directed retrieval is rewarded, undirected approaches are penalised.
- Evidence Programming's gap-directed retrieval should show the largest gains, because the system targets specific evidence types rather than issuing generic queries into a vast search space.
- connectomeDB\* should show clear differentiation since claims require finding specific experimental evidence in primary research articles — exactly the multi-hop evidence gathering Evidence Programming is designed for.

**Group 1 (constrained retrieval) — secondary diagnostic narrative:**
- If a system performs well in Group 1 but poorly in Group 2, the bottleneck is retrieval — the system can reason correctly when evidence is provided but cannot find it autonomously.
- If a system performs poorly in both groups, the bottleneck is reasoning — even perfect evidence does not help.
- Evidence Programming should be the only system that maintains strong performance in both groups, validating that its gains in Group 2 come from both better retrieval (gap-directed queries) and better reasoning (structured evidence state), not from retrieval alone.
- The step-wise progression (Random → LLM-only → Fixed-k RAG → ReAct → Factcheck-GPT → SAFE → FIRE → Evidence Programming) illustrates each component's contribution to reasoning quality in isolation from retrieval.

---

## 9. Excluded Systems (With Justification)

| System | Reason for Exclusion |
|--------|---------------------|
| **Self-RAG** (ICLR 2024) | Requires fine-tuned Llama-2 with custom reflection tokens. Cannot fairly compare with a different backbone LLM. Simulating via prompting is not truly Self-RAG. |
| **MultiVerS** (NAACL 2022) | Pre-LLM supervised classifier (Longformer). Tests nothing about our contributions. Incompatible task framing (requires pre-selected abstracts). SciFact-only. |
| **BOOST / ProgramFC** (ACL 2023, arXiv 2025) | Generate reasoning programs (meta-rules, second-order logic). Focuses on the reasoning decomposition problem, not iterative evidence gathering. |
| **PCC** (arXiv 2026) | Withdrawn from ICLR 2026. Not published at a peer-reviewed venue. |
| **COVID-19 SRAG** (JMIR 2025) | Scoped exclusively to COVID-19 claims. Not a general scientific verification system. |
| **Tool-MAD / DelphiAgent** | Multi-agent debate systems for general fact-checking. Substantial re-engineering needed for PubMed-based scientific verification. |
| **IRCoT** (NeurIPS 2023) | Overlaps with ReAct as an adaptive prompting strategy. Forces retrieval at every CoT step (a constraint that may be suboptimal). ReAct is more general, more well-known, and scales better with larger models. |

---

## 10. Results Table Templates

### Group 1: Constrained Retrieval + Verdict

```latex
\begin{table}[ht]
\centering
\small
\caption{\textbf{Group 1: Constrained retrieval evaluation.}
All systems retrieve from the same corpus.
Pair-level Label F1 (\%). Best in \textbf{bold}.
Mean over 3 runs.}
\begin{tabular}{l c | c c c | c c c}
\toprule
 & & \multicolumn{3}{c|}{\textbf{SciFact-Open}}
 & \multicolumn{3}{c}{\textbf{CIViC-Fact}} \\
\textbf{Method} & \textbf{Model}
 & \textbf{F1} & \textbf{Cost} & $\Delta$
 & \textbf{F1} & \textbf{Cost} & $\Delta$ \\
\midrule
\multicolumn{8}{l}{\textit{No retrieval}} \\
Random & -- & & & & & & \\
LLM-only & x & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{Static retrieval}} \\
LLM + 5 search results & x & & & & & & \\
Fixed-5 RAG (BM25) & x & & & & & & \\
Fixed-10 RAG (BM25) & x & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{Prompting strategy}} \\
ReAct & x & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{Specialised RAG}} \\
OpenScholar & OS-8B & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{LLM verification pipelines}} \\
Factcheck-GPT & x & & & & & & \\
SAFE & x & & & & & & \\
FIRE & x & & & & & & \\
\midrule
\textbf{Evidence Prog. (ours)} & x & & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

### Group 2: Open Retrieval + Claim-Level Verdict

```latex
\begin{table}[ht]
\centering
\small
\caption{\textbf{Group 2: Open retrieval, claim-level evaluation.}
Systems search freely. Claim-level F1 (\%).
Best in \textbf{bold}. Mean over 3 runs.}
\begin{tabular}{l c | c c c | c c c}
\toprule
 & & \multicolumn{3}{c|}{\textbf{connectomeDB*}}
 & \multicolumn{3}{c}{\textbf{SIGNOR*}} \\
\textbf{Method} & \textbf{Model}
 & \textbf{F1} & \textbf{Cost} & $\Delta$
 & \textbf{F1} & \textbf{Cost} & $\Delta$ \\
\midrule
\multicolumn{8}{l}{\textit{No retrieval}} \\
Random & -- & & & & & & \\
LLM-only & x & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{Static retrieval}} \\
LLM + 5 search results & x & & & & & & \\
Fixed-5 RAG (BM25) & x & & & & & & \\
Fixed-10 RAG (BM25) & x & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{Prompting strategy}} \\
ReAct & x & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{Specialised RAG}} \\
OpenScholar & OS-8B & & & & & & \\
\midrule
\multicolumn{8}{l}{\textit{LLM verification pipelines}} \\
Factcheck-GPT & x & & & & & & \\
SAFE & x & & & & & & \\
FIRE & x & & & & & & \\
\midrule
\textbf{Evidence Prog. (ours)} & x & & & & & & \\
\bottomrule
\end{tabular}
\end{table}
```
