## Summary for Abstract Figure — Metacognitive Evidence Verification of Gene Regulatory Networks

### What the system does

The system takes **Gene Regulatory Network (GRN) edges** from the SIGNOR database — each a (source gene, target gene, interaction type) triple — and verifies whether each edge is supported by scientific literature, using an LLM-in-the-loop metacognitive control loop.

### Concrete running example (use this in the figure)

**Input edge:** `(TP53, BCL2L1, down-regulates activity)`
from the SIGNOR ground-truth dataset. This gets converted into:

> *"Does TP53 directly inhibit BCL2L1 (either through inhibition or destabilization)?"*

---

### Pipeline stages (left-to-right flow for the figure)

#### Stage 1 — GRN Edge Input
A directed graph edge from SIGNOR: **TP53 ⊣ BCL2L1** (down-regulates activity). The ground truth label is **true positive** (known validated edge). The dataset contains both true positive and true negative edges with SIGNOR schema columns (ENTITYA, ENTITYB, EFFECT, MECHANISM, PMID, SENTENCE).

#### Stage 2 — Natural Language Conversion
`construct_signor_question()` maps interaction types to natural language:
- `up-regulates activity` → *"Does SOURCE directly activate TARGET?"*
- `down-regulates quantity by destabilization` → *"Does SOURCE directly inhibit TARGET?"*

#### Stage 3 — Evidence Retrieval (PubMed)
`RelevancePubMedSearcher` queries PubMed ranked by relevance. For our example, the query yields papers like PMID:16455050 with the sentence: *"tp53 can directly bind bcl2, and consequently inhibit their anti-apoptotic activities."* Each result becomes a `PaperRecord(pmid, title, abstract)` stored in the `EvidenceState`.

#### Stage 4 — LLM Processing (two operations per paper)
1. **Summarize** — LLM condenses each abstract into a focused summary
2. **Extract Facts** — LLM extracts stance-labeled facts:
   - `Fact(text="p53 binds BCL-XL and inhibits its anti-apoptotic activity", stance=SUPPORT, source_pmid="16455050")`
   - `Fact(text="BCL-XL can function independently of p53 in some contexts", stance=NEUTRAL, source_pmid="...")`

The LLM follows the project's OpenAI-compatible client pattern — local models connect to `localhost:8000/v1`; cloud models (GLM-4.6, Claude) use their respective APIs.

#### Stage 5 — Sufficiency Classifier φ (the stopping criterion)
A **lightweight classifier** (not an LLM call) evaluates the evidence state:
- Input signals: number of papers, number of facts, SUPPORT/REFUTE fractions
- Output: `{label: SUFFICIENT_SUPPORT, confidence: 0.85, gaps: []}`
- Decision rule (backbone heuristic): if ≥3 facts and ≥70% agree → SUFFICIENT

This is a **drop-in interface** — the heuristic will be replaced by a trained MLP with 16 features (coverage, conflicts, LLM confidence signals, embedding similarity) without changing any calling code.

#### Stage 6 — Metacognitive Control Loop (Algorithm 1)
The core loop implements *metacognitive awareness* — the system knows what it knows and what it doesn't:

```
for t = 1 to T:
    (confidence, label, gaps) ← φ(state)      # assess sufficiency
    if confidence ≥ τ: return verdict           # early stop
    if gaps = ∅ and t > T/2: return INSUFFICIENT
    retrieve more papers for identified gaps     # gap-directed retrieval
    process new papers (summarize + extract)
    if token_count > B: compress(state)          # sufficiency-preserving compression
```

Key parameters: threshold $\tau = 0.7$, max iterations $T = 3$, context budget $B = 50{,}000$ tokens.

#### Stage 7 — Sufficiency-Preserving Compression (primary technical novelty)
When the evidence state exceeds the token budget, compression applies while preserving decision quality:

- **L1 (lossless):** Deduplicate facts with identical text + stance — implemented in backbone
- **L2 (lossy, guarded):** LLM-based synthesis refresh — deferred
- **L3 (aggressive, guarded):** Heavy synthesis — deferred
- **Invariant:** $|\phi(S') - \phi(S)| \leq \varepsilon$ — compression only proceeds if the classifier's confidence on the compressed state remains within $\varepsilon = 0.05$ of the original

#### Stage 8 — Structured Output
The final `StructuredOutcome` contains:
- `reasoning`: full chain-of-thought (evidence for Yes, evidence for No, final assessment)
- `answer`: `True` / `False` / `None` (maps to SUPPORT / REFUTE / INSUFFICIENT)
- `confidence`: from the sufficiency classifier
- `usage`: token statistics

**Example output for TP53 ⊣ BCL2L1:**
> *Evidence for YES: p53 directly binds BCL-XL, inhibiting its anti-apoptotic activity (PMID:16455050)...*
> *Answer: Yes* → verdict = **SUPPORT**, confidence = 0.85

---

### Figure layout suggestion

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  GRN Edge   │────▶│  NL Question │────▶│  PubMed Search  │────▶│  LLM Processing     │
│ TP53 ⊣ BCL2L1│    │"Does TP53    │     │ PMID:16455050   │     │ Summarize + Extract │
│ (SIGNOR DB) │     │ inhibit...?" │     │ PMID:18498746   │     │ Facts with stance   │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬──────────────┘
                                                                        │
                                                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                       Metacognitive Control Loop (Algorithm 1)                       │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐  │
│  │ Evidence     │───▶│ Sufficiency  │───▶│ Gap-Directed   │───▶│ Compression      │  │
│  │ State S      │    │ Classifier φ │    │ Retrieval      │    │ L1: dedup        │  │
│  │ papers,facts │    │ conf ≥ τ?    │    │ (if INSUFF.)   │    │ |φ(S')-φ(S)|≤ε  │  │
│  └──────────────┘    └──────────────┘    └────────────────┘    └──────────────────┘  │
│                              │ YES                                                    │
│                              ▼                                                        │
│                    ┌──────────────────┐                                                │
│                    │ VERDICT + REPORT │                                                │
│                    │ SUPPORT, p=0.85  │                                                │
│                    └──────────────────┘                                                │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Key terms for the figure labels

| Term | Meaning |
|---|---|
| **GRN edge** | Directed (source, target, effect) triple from SIGNOR |
| **Sufficiency classifier φ** | Lightweight model that predicts if evidence is enough to decide |
| **Sufficiency-preserving compression** | Compress evidence while guaranteeing $|\phi(S') - \phi(S)| \leq \varepsilon$ |
| **Gap-directed retrieval** | Retrieve papers targeting specific identified evidence deficiencies |
| **Metacognitive control** | The system monitors its own evidence state to decide when to stop |
| **Stance-labeled facts** | Extracted claims labeled SUPPORT / REFUTE / NEUTRAL per paper |
| **EvidenceState** | Mutable container holding papers, summaries, and facts |

### Second concrete example (for diversity in the figure)

**Input edge:** `(KRAS, RAF1, up-regulates)` — a well-known oncogenic signaling edge.

Question: *"Does KRAS directly activate RAF1 (either through activation or increase of expression)?"*

PubMed returns PMID:16293107 with: *"active RAS binds and activates the RAF kinase..."*

Extracted fact: `Fact("RAS-GTP binds RAF1 effector region, inducing membrane translocation and kinase activation", stance=SUPPORT, pmid="16293107")`

After 1 iteration the classifier returns `{label: SUFFICIENT_SUPPORT, confidence: 0.90}` → early stop, verdict = **SUPPORT**.