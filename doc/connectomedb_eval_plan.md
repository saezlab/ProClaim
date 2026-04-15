# ConnectomeDB Evaluation Implementation Plan

## Overview

Adapt the existing SIGNOR batch evaluation pipeline for the ConnectomeDB dataset.

**Dataset**: `/hps/nobackup/saezrodriguez/shared_datasets/claims/datasets/connectomedb.csv`
- 547 data rows (548 lines including header)
- Claims are **pre-formed** in the `claim` column — no construction needed
- No flip variants — each claim is run once (no forward/flipped pairs)
- Labels: `SUPPORT` / `REFUTE` / `UNCERTAIN`

**Key differences from SIGNOR:**

| | SIGNOR | ConnectomeDB |
|---|---|---|
| Data rows | 67 | 547 |
| Claim source | Constructed from ENTITYA/B/EFFECT | Pre-formed in `claim` column |
| Flip variants | Yes (for SUPPORTED rows) | None |
| Runs per row (3 reps) | Up to 6 (with flip) | 3 |
| Parallel chunks | 4 | **8** |
| SLURM array | `--array=0-3` | `--array=0-7` |

---

## Files to Create / Modify

### 1. `experiments/run_connectomedb_eval.py` *(new)*

Adapted from `experiments/run_signor_eval.py`. Key changes:

**Remove:**
- `construct_signor_claim()` — not needed; claim is taken directly from `row["claim"]`
- `get_flipped_label()` — no flip logic
- `flip_variants` loop — always just `[False]`
- SIGNOR-specific column references (`ENTITYA`, `ENTITYB`, `EFFECT`, `SIGNOR_ID`)

**Change input path default:**
```python
DEFAULT_INPUT_CSV = "/hps/nobackup/saezrodriguez/shared_datasets/claims/datasets/connectomedb.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "connectomedb_eval_results.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "connectomedb_eval"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "connectomedb_eval_config.yaml"
```

**Output columns** (replacing SIGNOR-specific columns):
```python
out_cols = [
    "CDB_ID", "Ligand", "Receptor", "Label",
    "Claim_String", "Repetition", "Agent_Verdict",
    "Agent_Confidence", "Reasoning_Snippet", "Output_Directory",
    "Input_Tokens", "Output_Tokens", "Cache_Creation_Tokens", "Cache_Read_Tokens",
    "Total_Input_Tokens", "Cost_Estimate"
]
```

**Main loop** (simplified, no flip):
```python
for idx, row in df.iterrows():
    cdb_id = row.get("id", f"ROW_{idx}")
    ligand = row.get("ligand", "")
    receptor = row.get("receptor", "")
    label = row.get("label", "UNCERTAIN")
    claim_str = row["claim"]          # use directly, no construction

    for rep in range(1, args.reps + 1):
        if (cdb_id, rep) in existing_runs:
            continue

        run_dir = results_dir / cdb_id / f"rep_{rep}"
        stats = run_evaluation(claim_str, run_dir, config_path, ...)

        # write row to CSV...
```

**Resume-key** in `existing_runs`:
```python
existing_runs.add((row.get("CDB_ID"), int(row.get("Repetition", 1))))
```

**Run directory naming** — note: `cdb_id` values like `CDB25:0003056` contain colons, which are invalid in filesystem paths. Sanitize:
```python
safe_id = cdb_id.replace(":", "_").replace(" ", "_")
run_dir = results_dir / safe_id / f"rep_{rep}"
```

---

### 2. `experiments/configs/connectomedb_eval_config.yaml` *(new)*

Copy `signor_eval_config.yaml`, change only the output_dir:

```yaml
# ConnectomeDB Evaluation Config
mode: sdk
max_iterations: 4
sufficiency_threshold: 0.80

llm:
  model: claude-sonnet-4-6
  agent_base_url: "https://api.anthropic.com"
  subagent_model: qwen3.5-9b
  temperature: 0.7

sufficiency_backend: haiku

output_dir: results/connectomedb_eval_prompt
mlp_model_dir: results/ablation/models/tau_0.50_seed_42

verbose: True
```

---

### 3. `scripts/run_connectomedb_batch.sh` *(new)*

Copy `scripts/run_signor_batch.sh` verbatim, changing only:
- The echo header text
- The python script call: `run_connectomedb_eval.py`
- The config path: `connectomedb_eval_config.yaml`
- The results path in the completion message

```bash
uv run python experiments/run_connectomedb_eval.py \
    --config experiments/configs/connectomedb_eval_config.yaml \
    "$@"
```

---

### 4. `scripts/slurm_connectomedb_job.sh` *(new)*

Copy `scripts/slurm_signor_job.sh`, changing:

**Header / job name references:**
```bash
# Replace "SIGNOR" → "ConnectomeDB" in echo strings
```

**8-chunk row splits** (547 rows ÷ 8 = 68–69 rows each):

```bash
case "$TASK_ID" in
    0) ROW_START=0;   ROW_LIMIT=69 ;;
    1) ROW_START=69;  ROW_LIMIT=69 ;;
    2) ROW_START=138; ROW_LIMIT=69 ;;
    3) ROW_START=207; ROW_LIMIT=69 ;;
    4) ROW_START=276; ROW_LIMIT=68 ;;
    5) ROW_START=344; ROW_LIMIT=68 ;;
    6) ROW_START=412; ROW_LIMIT=68 ;;
    7) ROW_START=480; ROW_LIMIT=67 ;;   # 480 + 67 = 547
    *) ROW_START=0;   ROW_LIMIT=0  ;;
esac
```

> **Note**: Chunk 7 gets 67 rows (not 68) because the total is 547:
> 3×69 + 4×68 + 1×67 = 207 + 272 + 67 = 546. That's off by one.
> Correct split: first **3 chunks** get 69, remaining **5 chunks** get 68:
> 3×69 + 5×68 = 207 + 340 = 547 ✓

```bash
case "$TASK_ID" in
    0) ROW_START=0;   ROW_LIMIT=69 ;;
    1) ROW_START=69;  ROW_LIMIT=69 ;;
    2) ROW_START=138; ROW_LIMIT=69 ;;
    3) ROW_START=207; ROW_LIMIT=68 ;;
    4) ROW_START=275; ROW_LIMIT=68 ;;
    5) ROW_START=343; ROW_LIMIT=68 ;;
    6) ROW_START=411; ROW_LIMIT=68 ;;
    7) ROW_START=479; ROW_LIMIT=68 ;;   # 479 + 68 = 547 ✓
    *) ROW_START=0;   ROW_LIMIT=0  ;;
esac
```

**Output CSV path:**
```bash
OUTPUT_CSV="${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}/results_chunk${TASK_ID}.csv"
```

**Batch runner call:**
```bash
bash "${PROJECT_ROOT}/scripts/run_connectomedb_batch.sh" ${RUN_ARGS} ${EXTRA_ARGS:-}
```

---

### 5. `scripts/submit_connectomedb_batch.sh` *(new)*

Copy `scripts/submit_signor_batch.sh`, changing:

**Variables:**
```bash
JOB_NAME="connectomedb-eval"
JOB_SCRIPT="${PROJECT_ROOT}/scripts/slurm_connectomedb_job.sh"
JOB_INFO_FILE="${PROJECT_ROOT}/.connectomedb_job_info"
```

**Array size — 8 chunks instead of 4:**
```bash
ARRAY_FLAG="--array=0-7"
```

**Merge job** — concatenate 8 chunk CSVs (same logic, just more files):
```bash
RESULTS_DIR="${PROJECT_ROOT}/results/connectomedb_eval_${RUN_TAG}"
# --wrap python glob pattern stays the same; it will find results_chunk0..7.csv
```

**Help text updates:**
- `(default)` mode: `8-chunk parallel: 8 GPUs, auto-merged results`
- `--limit` note: `default: all 547`

---

## Chunk Layout Summary

| Task ID | Row start | Row count | Rows |
|---------|-----------|-----------|------|
| 0 | 0 | 69 | 0–68 |
| 1 | 69 | 69 | 69–137 |
| 2 | 138 | 69 | 138–206 |
| 3 | 207 | 68 | 207–274 |
| 4 | 275 | 68 | 275–342 |
| 5 | 343 | 68 | 343–410 |
| 6 | 411 | 68 | 411–478 |
| 7 | 479 | 68 | 479–546 |
| **Total** | | **547** | |

---

## Execution

```bash
# Default: 8-GPU parallel (recommended)
bash scripts/submit_connectomedb_batch.sh

# Test single claim
bash scripts/submit_connectomedb_batch.sh --single --limit 1

# Check status
bash scripts/submit_connectomedb_batch.sh --status

# Follow logs
bash scripts/submit_connectomedb_batch.sh --logs

# Download results
scp wuy@ihpc.ebi.ac.uk:/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct/results/connectomedb_eval_<RUN_TAG>/results.csv .
```

---

## File Summary

| File | Action | Notes |
|------|--------|-------|
| `experiments/run_connectomedb_eval.py` | **New** | No flip, reads `claim` column directly, sanitize `:` in IDs for paths |
| `experiments/configs/connectomedb_eval_config.yaml` | **New** | Copy of signor config, updated output_dir |
| `scripts/run_connectomedb_batch.sh` | **New** | Calls `run_connectomedb_eval.py` |
| `scripts/slurm_connectomedb_job.sh` | **New** | 8-chunk splits, `--array=0-7` |
| `scripts/submit_connectomedb_batch.sh` | **New** | Entry point, 8 GPUs default |

No existing SIGNOR files are modified.
