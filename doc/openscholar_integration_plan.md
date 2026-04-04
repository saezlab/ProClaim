# OpenScholar Integration Plan

## Overview

OpenScholar (Asai et al., Nature 2025) is included in the evaluation as a **specialised RAG** baseline. It contributes two architectural innovations over standard RAG: (1) dense retrieval over a large scientific corpus (45M papers from S2ORC), and (2) an iterative self-feedback generation loop where the model critiques its own synthesis and issues follow-up queries.

Two integration tracks are planned:

| Track | Label in paper | Model | Infrastructure | Role |
|-------|---------------|-------|----------------|------|
| **OS-GPT** | OpenScholar (GPT-4o) | Same backbone LLM as all other baselines | API keys only | Main results table |
| **OS-8B** | OpenScholar-8B | Llama 3.1 8B fine-tuned for scientific synthesis | GPU + ~500 GB storage | Supplementary row |

**OS-GPT is completed first** — it unblocks evaluation immediately and provides a fair architectural comparison with a controlled model. OS-8B is blocked on storage/GPU availability and goes in supplementary results.

---

## OS-8B vs OS-GPT: Differences

| Dimension | OS-8B | OS-GPT |
|-----------|-------|--------|
| **Model** | Llama 3.1 8B fine-tuned on scientific synthesis | Backbone LLM (e.g. GPT-4o, GLM) |
| **Retrieval** | Dense retrieval over own 45M-paper datastore (~500 GB) | Semantic Scholar public API |
| **Reranking** | Trained cross-encoder reranker | None (S2 relevance ranking) |
| **Self-feedback** | Built into fine-tuned model weights | Prompted via standard chat API |
| **Faithfulness to paper** | Full reproduction | Approximation — preserves loop architecture, not weights |
| **Model comparability** | Unfair (different family + size) | Fair (same backbone as all baselines) |
| **What it tests** | Whether a specialised system beats our architecture | Whether OS's *loop design* alone helps vs. our system |
| **Infrastructure** | GPU node + 500 GB storage + HuggingFace models | API keys only |
| **Implementation effort** | High (~2 days) | Low (~3 hrs) |

**Paper treatment:**
- OS-GPT goes in the **main results table** — same backbone LLM ensures architectural isolation.
- OS-8B goes in a **supplementary row or footnote** — model family difference must be acknowledged explicitly.

---

## System Description

Both tracks implement the same conceptual loop:

```
Round 0:  retrieve k=10 papers
Round 1:  synthesize claim evidence → identify retrieval gaps → issue follow-up queries
Round 2:  retrieve k=5 papers per gap query → re-synthesize
Final:    extract structured verdict via shared verification prompt
```

The only structural difference is *where* retrieval and synthesis happen (local 8B model + dense datastore vs. S2 API + backbone LLM prompting).

---

## Track 1: OS-GPT (API-only Approximation)

### Step 1 — Confirm Semantic Scholar API access

```bash
# Add to .env
SEMANTIC_SCHOLAR_API_KEY=...
# Free tier: 1 req/sec. Recommended: request an API key for 100 req/sec.
```

### Step 2 — Add `semantic_scholar_search` to shared retrieval

Add to `experiments/baselines/shared/retrieval.py`:

```python
def semantic_scholar_search(query: str, k: int = 10) -> list[dict]:
    """Search Semantic Scholar /graph/v1/paper/search.
    Returns list of {paperId, title, abstract} dicts."""
```

This function is also used by the ReAct and FIRE baselines for Group 2 open-retrieval evaluation.

### Step 3 — Implement `experiments/baselines/openscholar_gpt.py`

Three helper LLM calls per round (all use the shared `LLMBackend`):

1. `_synthesize(claim, papers)` — draft a synthesis paragraph from current papers
2. `_identify_gaps(claim, synthesis)` — list retrieval gaps as follow-up query strings; return `[]` if synthesis is sufficient
3. `_extract_verdict(claim, synthesis)` — shared verification prompt → `SUPPORT` / `REFUTE` / `NEI`

```python
class OpenScholarGPT:
    name = "openscholar_gpt"

    def __init__(self, llm: LLMBackend, max_rounds: int = 3, k: int = 10):
        self.llm = llm
        self.max_rounds = max_rounds
        self.k = k

    def verify(self, claim_id: str, claim: str, gold_label: str) -> BaselineResult:
        papers = semantic_scholar_search(claim, k=self.k)
        synthesis = self._synthesize(claim, papers)

        for _ in range(self.max_rounds - 1):
            gaps = self._identify_gaps(claim, synthesis)
            if not gaps:
                break
            for gap_query in gaps[:2]:       # cap follow-up queries per round
                papers += semantic_scholar_search(gap_query, k=5)
            synthesis = self._synthesize(claim, papers)

        verdict = self._extract_verdict(claim, synthesis)
        return BaselineResult(
            claim_id=claim_id,
            claim=claim,
            gold_label=gold_label,
            predicted_label=verdict.label,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            evidence=[p["paperId"] for p in papers],
            baseline_name=self.name,
        )
```

### Step 4 — Register in `run_baselines_datasets.py`

```python
elif name == "openscholar_gpt":
    from baselines.openscholar_gpt import OpenScholarGPT
    return OpenScholarGPT(llm=llm, max_rounds=3, k=10)
```

### Step 5 — Smoke-test on 5 SIGNOR* claims

Check output format, cost per claim (expect ~15–25 LLM calls for 3 rounds × 3 calls + overhead), and that verdict parsing produces valid labels before a full run.

---

## Track 2: OS-8B (Full Reproduction)

### Step 1 — Check HPC storage availability

```bash
df -h /hps/nobackup/saezrodriguez/
# Need ~500 GB free for datastore + ~20 GB for model weights
```

Abort this track if storage is unavailable — do not compress or subset the datastore, as it breaks retrieval quality.

### Step 2 — Reserve a GPU node

```bash
salloc --gres=gpu:1 --mem=40G --time=24:00:00
```

OpenScholar-8B requires a single GPU with ≥24 GB VRAM for inference at batch size 1.

### Step 3 — Clone and install

```bash
git clone https://github.com/AkariAsai/OpenScholar
cd OpenScholar
pip install -e ".[gpu]"
```

### Step 4 — Download artefacts

```bash
# Datastore (~500 GB) — verify exact size on their HuggingFace page before starting
python download_datastore.py \
    --output /hps/nobackup/saezrodriguez/ail/openscholar_datastore

# Model weights (~16 GB)
huggingface-cli download OpenScholar/OpenScholar-8B \
    --local-dir /hps/nobackup/saezrodriguez/ail/openscholar-8b

# Trained reranker (~1 GB)
huggingface-cli download OpenScholar/reranker \
    --local-dir /hps/nobackup/saezrodriguez/ail/openscholar-reranker
```

### Step 5 — Verify pipeline on one claim

```bash
python run_openscholar.py \
    --claim "MAPK1 directly activates H3-3A." \
    --datastore_path /hps/nobackup/saezrodriguez/ail/openscholar_datastore \
    --model_path /hps/nobackup/saezrodriguez/ail/openscholar-8b \
    --reranker_path /hps/nobackup/saezrodriguez/ail/openscholar-reranker
```

Confirm the synthesis text is returned and reasonable before writing the wrapper.

### Step 6 — Implement `experiments/baselines/openscholar_8b.py`

Key design decisions:
- Wrap their pipeline call inside `verify(claim_id, claim, gold_label)`.
- **Claim framing:** `"What does the scientific literature say about the following claim? Is it supported or refuted?\n\nClaim: {claim}"` — OpenScholar is designed for synthesis questions, not binary verdicts.
- **Verdict extraction:** after OS-8B produces its synthesis, make **one additional call to the shared backbone LLM** using the shared verification prompt to extract a structured verdict. This keeps the final verdict step comparable across all baselines.
- **Cost tracking:** log OS-8B inference wall-time and token-equivalent separately from the backbone LLM verdict-extraction cost. These are two separate cost columns in the results table.

```python
class OpenScholar8B:
    name = "openscholar_8b"

    def __init__(
        self,
        datastore: str,
        model_path: str,
        reranker_path: str,
        verdict_llm: LLMBackend,   # backbone LLM for final verdict step only
    ):
        ...

    def verify(self, claim_id: str, claim: str, gold_label: str) -> BaselineResult:
        framed = f"What does the scientific literature say about the following claim? "
               f"Is it supported or refuted?\n\nClaim: {claim}"
        synthesis = self._run_os_pipeline(framed)          # calls OS-8B + datastore
        verdict = self.verdict_llm.extract_verdict(claim, synthesis)   # shared prompt
        return BaselineResult(...)
```

### Step 7 — Slurm job script

OS-8B cannot run on the same CPU-only job as other baselines. Create a dedicated Slurm script `scripts/run_openscholar_8b.sh` with GPU resource requests, separate from `run_llm_baselines_slurm.sh`.

### Step 8 — Register in `run_baselines_datasets.py`

```python
elif name == "openscholar_8b":
    from baselines.openscholar_8b import OpenScholar8B
    return OpenScholar8B(
        datastore=args.os_datastore,
        model_path=args.os_model,
        reranker_path=args.os_reranker,
        verdict_llm=llm,
    )
```

Add `--os-datastore`, `--os-model`, `--os-reranker` CLI arguments to `run_baselines_datasets.py` (only parsed when `--baseline openscholar_8b`).

---

## Corpus Coverage Notes

| Dataset | OS-8B datastore | OS-GPT (S2 API) | Action |
|---------|----------------|----------------|--------|
| SciFact-Open | S2ORC subset — good overlap | S2 API — good | None |
| CIViC-Fact | Clinical oncology — covered | S2 API — covered | None |
| SIGNOR* | Intracellular signaling — mostly covered | S2 API — mostly covered | Flag recent papers (post-datastore snapshot) in discussion |
| ConnectomeDB* | Some niche ligand-receptor pairs | S2 API — may miss very recent papers | Same flag |

For Group 2 open-retrieval evaluation, note in the paper that OS-8B's datastore has a fixed snapshot date. Papers published after that date are not accessible to OS-8B, creating a systematic retrieval disadvantage for recent SIGNOR* and ConnectomeDB* claims. OS-GPT via live S2 API does not have this limitation.

---

## Fairness Checklist

- [ ] Final verdict extraction for OS-8B uses the **shared verification prompt** (same as all other baselines)
- [ ] OS-8B model family difference (Llama 3.1 8B) acknowledged in paper and results table
- [ ] OS-8B inference cost and backbone LLM verdict-extraction cost tracked separately
- [ ] OS-GPT self-feedback iteration count logged as "LLM calls/claim" in cost accounting
- [ ] Datastore snapshot date noted relative to SIGNOR* and ConnectomeDB* paper publication dates
- [ ] OS-GPT uses the same `LLMBackend` instance as all non-specialised baselines (same model, temperature=0)
- [ ] `max_rounds=3` for OS-GPT — set so median token usage is comparable to FIRE and SAFE

---

## Implementation Order

1. **OS-GPT first** — unblocks the main results table, no infrastructure dependencies
2. **OS-8B after** — only when GPU node and storage are confirmed available; results go in supplementary
