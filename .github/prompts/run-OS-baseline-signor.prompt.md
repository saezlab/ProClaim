---
description: "Run one or more SIGNOR claims through OpenScholar (Claude Sonnet 4.6) and print structured verdicts"
agent: agent
argument-hint: "Optional: single index (e.g. 5), inclusive range (e.g. 0-9), comma-separated list (e.g. 0,2,5), or 'all' (default: 0)"
tools: [run_in_terminal, read_file, create_file]
---

Run one or more SIGNOR claims through OpenScholar using Claude Sonnet 4.6 and print structured
verdicts. OpenScholar is instructed via `--task_name claim_verdict` to emit JSON directly.
Follow every step below in order.

### Parameters used

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--api` | `anthropic` | Routes through LiteLLM (`anthropic/claude-sonnet-4-6`) |
| `--model_name` | `claude-sonnet-4-6` | Anthropic Claude Sonnet 4.6 |
| `--task_name` | `claim_verdict` | Custom task in `src/instructions.py` — outputs structured JSON verdict |
| `--zero_shot` | *(flag)* | No few-shot demonstrations prepended to the prompt |
| `--top_n` | `10` | Max passages forwarded to the generator |
| `--max_tokens` | *(omitted)* | No constraint — OpenScholar default of 3000 applies |
| `--ss_retriever` | *(flag)* | Queries Semantic Scholar for supporting passages |
| `--feedback` | *(flag)* | Self-reflective loop: keyword extraction → S2 search → rerank → answer edit (1–3 extra LLM calls) |
| `--use_contexts` | **not set** | No pre-retrieved passages injected; S2 retrieval fills context |
| `temperature` | `0.7` | Hardcoded in `OpenScholar.generate_response()` |

**Not used in this configuration:** `--posthoc_at`, `--use_contexts`, `--llama3`, `--norm_cite`

> **Note:** `--feedback` + `--ss_retriever` will call the reranker when new S2 papers are found. Always include `--ranking_ce --reranker OpenScholar/OpenScholar_Reranker` when using `--ss_retriever --feedback` — omitting `--reranker` causes a crash (`AttributeError: 'NoneType'.compute_score`).

### Results locations

| Artefact | Path |
|----------|------|
| Input JSONL | `/tmp/signor_claims.jsonl` |
| Raw OpenScholar output | `/tmp/signor_claims_output.json` |
| Canonical results JSONL | `grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.jsonl` |
| Aggregated metrics | `grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_metrics.json` |
| Run log | `grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.log` |

---

## Step 1 — Pick the claims

Read the SIGNOR dataset:

```
/path_to/claim_datasets/datasets/signor.csv
```

Determine the set of rows to process from the argument `${{ index | default: 0 }}` (0-based, header excluded):

| Argument form | Example | Rows selected |
|---------------|---------|---------------|
| Single integer | `5` | Row 5 only |
| Inclusive range | `0-9` | Rows 0 through 9 |
| Comma-separated list | `0,2,5` | Rows 0, 2 and 5 |
| `all` | `all` | Every row |
| *(omitted)* | — | Row 0 (default) |

For each selected row extract the `claim` and `label` columns.

Frame each claim as the OpenScholar input question:

```
What does the scientific evidence say about {claim}?
Is it supported or refuted by the literature?
```

---

## Step 2 — Write the input JSONL

Create `/tmp/signor_claims.jsonl` with one record per selected claim (one JSON object per line):

```json
{"input": "<framed question — claim 0>", "ctxs": [], "answer": ""}
{"input": "<framed question — claim 1>", "ctxs": [], "answer": ""}
```

`ctxs: []` triggers live S2 retrieval inside OpenScholar (no `--use_contexts` flag).

---

## Step 3 — Run OpenScholar

```sh
cd /path_to/OpenScholar
source .venv/bin/activate

# Keys – S2 allows anonymous (rate-limited) access; empty string avoids the KeyError crash
export S2_API_KEY=""
export ANTHROPIC_API_KEY="$(grep ANTHROPIC_API_KEY /path_to/grn-llm-correct/.env | cut -d= -f2)"

python run.py \
  --input_file /tmp/signor_claims.jsonl \
  --output_file /tmp/signor_claims_output.json \
  --api anthropic \
  --model_name claude-sonnet-4-6 \
  --task_name claim_verdict \
  --zero_shot \
  --top_n 10 \
  --ss_retriever \
  --feedback \
  --ranking_ce \
  --reranker OpenScholar/OpenScholar_Reranker
```

`--ss_retriever` triggers Semantic Scholar keyword retrieval; `--feedback` enables the
self-reflective loop (keyword extraction → S2 search → rerank → answer edit, 1–3 extra
LLM calls). `--ranking_ce --reranker` is required with `--ss_retriever --feedback`;
omitting `--reranker` causes a crash (`AttributeError: 'NoneType'.compute_score`).
`--max_tokens` is omitted — OpenScholar's default of 3000 applies.

---

## Step 4 — Convert output, compute metrics, and produce run log

This step converts the raw OpenScholar output into the canonical `BaselineResult` JSONL
format used by all baselines in `grn-llm-correct/experiments/`, computes aggregated metrics,
and writes a human-readable run log.

### 4a — Convert to canonical JSONL

Read `/tmp/signor_claims_output.json` and the SIGNOR CSV. For each item the `output` field
contains the raw JSON verdict from OpenScholar. Parse all items and emit one
`BaselineResult`-compatible JSON object per line to the canonical results JSONL.

The JSONL schema must match `grn-llm-correct/experiments/baselines/shared/verdict.py`
(`BaselineResult`). Every record has exactly these fields:

```json
{
  "claim_id": "<id column from CSV>",
  "claim": "<claim column from CSV>",
  "gold_label": "SUPPORT|REFUTE|NEI",
  "predicted_label": "SUPPORT|REFUTE|NEI",
  "confidence": 0.0,
  "reasoning": "<verdict.reasoning>",
  "evidence": ["[0]", "[1]"],
  "input_tokens": 0,
  "output_tokens": 0,
  "cost_usd": 0.017316,
  "latency_seconds": 53.96,
  "baseline_name": "open_scholar",
  "model": "claude-sonnet-4-6",
  "dataset": "signor"
}
```

Normalise labels using the map:
`SUPPORTS/SUPPORTED → SUPPORT`, `CONTRADICTS/REFUTES/REFUTED/WRONG → REFUTE`, `NEI/UNCERTAIN → NEI`.

Parse each OpenScholar item:

```python
import csv, json
from collections import defaultdict

rows = []
with open("/path_to/claim_datasets/datasets/signor.csv") as f:
    for row in csv.DictReader(f):
        rows.append(row)

with open("/tmp/signor_claims_output.json") as f:
    data = json.load(f)
items = data if isinstance(data, list) else data["data"]

def normalize(label):
    label = label.upper().strip()
    if label in ("SUPPORTS", "SUPPORTED", "SUPPORT"):
        return "SUPPORT"
    if label in ("CONTRADICTS", "REFUTES", "REFUTED", "WRONG", "REFUTE"):
        return "REFUTE"
    return "NEI"

results = []
for row, item in zip(rows, items):
    try:
        verdict = json.loads(item["output"])
    except (json.JSONDecodeError, KeyError, TypeError):
        verdict = {"label": "NEI", "reasoning": item.get("output", ""), "evidence": [], "confidence": 0.0}
    results.append({
        "claim_id": row["id"],
        "claim": row["claim"],
        "gold_label": normalize(row["label"]),
        "predicted_label": normalize(verdict.get("label", "NEI")),
        "confidence": float(verdict.get("confidence", 0.0)) if isinstance(verdict.get("confidence"), (int, float)) else 0.0,
        "reasoning": verdict.get("reasoning", ""),
        "evidence": verdict.get("evidence", []),
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": float(item.get("total_cost", 0.0)),
        "latency_seconds": float(item.get("elapsed", 0.0)),
        "baseline_name": "open_scholar",
        "model": "claude-sonnet-4-6",
        "dataset": "signor",
    })
```

Write to:
```
grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.jsonl
```

### 4b — Compute aggregated metrics

Compute metrics matching the `EvaluationHarness.metrics()` schema from
`grn-llm-correct/experiments/baselines/shared/evaluate.py`, then wrap in the
`aggregate_metrics()` format (each scalar → `{mean, std}` with `std: 0.0` for a single
repeat). The metrics JSON must contain exactly these top-level keys:

```
n, n_repeats, accuracy, macro_f1, macro_fpr, macro_fnr, weighted_fpr, weighted_fnr,
per_class (SUPPORT/REFUTE/NEI → precision/recall/f1), total_cost_usd
```

Write to:
```
grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_metrics.json
```

### 4c — Print and save run log

Print the summary below to the terminal **and** save it to:
```
grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.log
```

```
=== Command ===
python run.py \
  --input_file /tmp/signor_claims.jsonl \
  --output_file /tmp/signor_claims_output.json \
  --api anthropic \
  --model_name claude-sonnet-4-6 \
  --task_name claim_verdict \
  --zero_shot \
  --top_n 10 \
  --ss_retriever \
  --feedback \
  --ranking_ce \
  --reranker OpenScholar/OpenScholar_Reranker

=== Parameters ===
API:           anthropic
Model:         claude-sonnet-4-6
Task:          claim_verdict
Mode:          zero_shot, S2 adaptive retrieval (ss_retriever + feedback + ranking_ce + reranker)
top_n:         10
reranker:      OpenScholar/OpenScholar_Reranker
max_tokens:    3000 (OpenScholar default, --max_tokens omitted)
temperature:   0.7 (hardcoded in OpenScholar)

=== Per-claim results (N=<n>) ===
Row  Claim (truncated to 60 chars)                        Gold     Predicted  Match
---  ----------------------------------------------------  -------  ---------  -----
0    <claim text>                                          SUPPORT  SUPPORT    Yes
1    <claim text>                                          REFUTE   NEI        No
...

=== Aggregate metrics ===
Accuracy:      <n_correct>/<n> (<pct>%)
Macro F1:      <macro_f1>
Macro FPR:     <macro_fpr>
Macro FNR:     <macro_fnr>
Weighted FPR:  <weighted_fpr>
Weighted FNR:  <weighted_fnr>
Per-class:
  Class      Correct  Total  Precision  Recall  F1
  SUPPORT    <n>      <n>    <p>        <r>     <f>
  REFUTE     <n>      <n>    <p>        <r>     <f>
  NEI        <n>      <n>    <p>        <r>     <f>

Total cost:    $<sum of cost_usd across all items> USD
Total time:    <sum of latency_seconds>s  (avg <avg>s / claim)

=== Output saved to ===
JSONL:   grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.jsonl
Metrics: grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_metrics.json
Log:     grn-llm-correct/results/baselines/open_scholar/claude-sonnet-4-6/signor_seed100.log
```