# SIGNOR Batch Script Refactor Plan

## Summary of Changes

Three improvements to `submit_signor_batch.sh` (and related scripts):

1. **Single-GPU architecture** — vLLM and evaluation code share the same SLURM job and GPU
2. **Timestamped output directories** — prevent overwriting previous runs
3. **4-GPU parallel chunking** — split 67 CSV rows across 4 SLURM array tasks

---

## Change 1: Single-Job Architecture (vLLM + eval on same GPU)

### Problem with current design
- `submit_signor_batch.sh` allocates **two separate SLURM jobs**:
  1. `start_vllm_ihpc.sh` → `srun` inside tmux on **GPU #1** → runs `vllm_node_setup.sh` → writes `hostname:port:gpus` to `.vllm_server_info`
  2. `sbatch` → eval job on **GPU #2** → reads `.vllm_server_info` → connects to vLLM over HPC internal network
- The hostname-parsing (`cut -d: -f1`) from `.vllm_server_info` is only needed because the two jobs may be on **different nodes**
- Wastes 2 GPUs; fragile due to random node assignment

### New design: single `sbatch` job
- One `sbatch` job with 1× A100
- Within the job script:
  1. Run `vllm_node_setup.sh` **in the background** (it already handles: Singularity, port detection, model config, parser flags)
  2. Wait for `vllm_node_setup.sh` to write its `.vllm_server_info` file (same as current polling logic, but now it's local)
  3. Read the **port only** from the info file — use `localhost:{port}` (not the hostname, since both processes are on the same node)
  4. Poll `http://localhost:{port}/health` until vLLM is ready
  5. Export `LLM_BASE_URL=http://localhost:{port}/v1/` and run `run_signor_batch.sh`
  6. On exit: `trap` to kill the background vLLM process (or just let it die with the job)
- Completely eliminates hostname parsing concerns — `localhost` always works on the same node

### Files affected

| File | Change |
|------|--------|
| `scripts/submit_signor_batch.sh` | Replace Steps 1+2 (vLLM startup + wait loop) with a single `sbatch` of the new self-contained job script. Remove all `--no-vllm`/`SKIP_VLLM` logic and the vLLM server check |
| `scripts/slurm_signor_job.sh` *(new)* | Self-contained job script: starts vLLM in bg via `vllm_node_setup.sh` → polls health → runs eval → cleanup |
| `scripts/run_signor_batch.sh` | Remove `.vllm_server_info` reading; simply use `LLM_BASE_URL=http://localhost:${VLLM_PORT}/v1/` (port passed via env var from the job script) |

### New `slurm_signor_job.sh` (pseudocode)

```bash
#!/bin/bash
#SBATCH directives injected by submit_signor_batch.sh via --wrap or heredoc

PROJECT_ROOT="/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct"
LOCAL_INFO_FILE="${PROJECT_ROOT}/.vllm_server_info_${SLURM_JOB_ID}"   # unique per job

# 1. Start vLLM in background using existing node setup script
bash "${PROJECT_ROOT}/scripts/vllm_node_setup.sh" "$LOCAL_INFO_FILE" "qwen3.5-9b" &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null; rm -f $LOCAL_INFO_FILE" EXIT

# 2. Wait for info file to appear (reuse existing wait pattern)
until [[ -f "$LOCAL_INFO_FILE" ]]; do sleep 5; done
VLLM_PORT=$(cut -d: -f2 < "$LOCAL_INFO_FILE")    # only port needed; hostname ignored

# 3. Poll health endpoint (same as current start_vllm_ihpc.sh logic)
until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do sleep 5; done

# 4. Run evaluation
export LLM_BASE_URL="http://localhost:${VLLM_PORT}/v1/"
bash "${PROJECT_ROOT}/scripts/run_signor_batch.sh" $EXTRA_ARGS
```

---

## Change 2: Timestamped Output Directories

### Current behaviour
- Results CSV: `results/signor_eval_results.csv` (fixed path, no timestamp)
- Per-claim dirs: `results/signor_eval/{SIGNOR_ID}/flip_{flip}/rep_{rep}/`
- Re-running either overwrites or silently skips rows (resume logic) — can't distinguish different experiment runs

### New design
- Generate a run tag at job submission time: `RUN_TAG=$(date +%Y%m%d_%H%M%S)`
- Pass it to `run_signor_eval.py` via new `--run-tag` argument
- Output structure becomes:
  - Results CSV: `results/signor_eval_{RUN_TAG}/results.csv`
  - Per-claim dirs: `results/signor_eval_{RUN_TAG}/{SIGNOR_ID}/flip_{flip}/rep_{rep}/`
- Persist tag in `.signor_job_info` alongside job ID so `--status`/`--logs` can reference it

### Files affected

| File | Change |
|------|--------|
| `experiments/run_signor_eval.py` | Add `--run-tag STR` argument; replace hardcoded `results/signor_eval` path with `results/signor_eval_{run_tag}`; also derive default `--output-csv` from the run tag |
| `scripts/run_signor_batch.sh` | Accept and pass `--run-tag` through to `run_signor_eval.py` |
| `scripts/slurm_signor_job.sh` *(new)* | Receive `RUN_TAG` from env var set by `submit_signor_batch.sh` before submission |
| `scripts/submit_signor_batch.sh` | Generate `RUN_TAG=$(date +%Y%m%d_%H%M%S)` at submit time; export it into the job; save `{JOB_ID} {RUN_TAG}` to `.signor_job_info` |

---

## Change 3: 4-GPU Parallel Chunking via SLURM Job Array

### Claim count calculation

Ground truth CSV has **67 rows**:
| Label | Count | Flip? | Runs per row (×3 reps) | Subtotal |
|-------|-------|-------|------------------------|----------|
| SUPPORTED | 34 | forward + flipped | 6 | 204 |
| WRONG | 29 | forward only | 3 | 87 |
| UNCERTAIN | 4 | forward only | 3 | 12 |
| **Total** | **67** | | | **303** |

Target: 4 A100 GPUs, ~75 runs each → **~17 rows per chunk** (actual run count per chunk varies with SUPPORTED/non-SUPPORTED distribution, but 17-row chunks are the best even split available without resorting the CSV)

### Proposed row splits

| Chunk (array task) | Rows (0-indexed) | Row count |
|---------------------|-----------------|-----------|
| 0 | 0–16 | 17 |
| 1 | 17–33 | 17 |
| 2 | 34–50 | 17 |
| 3 | 51–66 | 16 |

### Implementation: SLURM job array

Use `sbatch --array=0-3` so SLURM assigns unique `$SLURM_ARRAY_TASK_ID` (0–3) to each task.

**Port isolation**: Since all 4 tasks may run on the same 4-GPU node, each task needs a unique port. Use: `VLLM_PORT=$((8000 + SLURM_ARRAY_TASK_ID))` — tasks get ports 8000, 8001, 8002, 8003. SLURM assigns separate `CUDA_VISIBLE_DEVICES` per task, so GPU isolation is automatic.

**Row range per task** (in `slurm_signor_job.sh`):
```bash
case "$SLURM_ARRAY_TASK_ID" in
    0) ROW_START=0;  ROW_LIMIT=17 ;;
    1) ROW_START=17; ROW_LIMIT=17 ;;
    2) ROW_START=34; ROW_LIMIT=17 ;;
    3) ROW_START=51; ROW_LIMIT=16 ;;
esac
```

**New `--row-start` argument needed in `run_signor_eval.py`**: Currently `--limit N` means "first N rows". Add `--row-start INT` (default 0) so that `df = df.iloc[row_start : row_start + limit]`.

**Output CSVs**: Each task writes to its own file:
`results/signor_eval_{RUN_TAG}/results_chunk{SLURM_ARRAY_TASK_ID}.csv`

Auto-merge: after all 4 array tasks complete, a fifth `sbatch` job (submitted with `--dependency=afterok:{ARRAY_JOB_ID}`) runs a simple concatenation step and writes the final `results/signor_eval_{RUN_TAG}/results.csv`. The header from chunk 0 is used; chunks 1–3 append without their header line.

### Files affected

| File | Change |
|------|--------|
| `experiments/run_signor_eval.py` | Add `--row-start INT` argument; slice DataFrame with `df.iloc[row_start:row_start+limit]` |
| `scripts/slurm_signor_job.sh` *(new)* | Read `SLURM_ARRAY_TASK_ID` → derive port offset and row range |
| `scripts/submit_signor_batch.sh` | **4-chunk parallel is now the default**; submit with `--array=0-3`; automatically submit a merge job with `--dependency=afterok`; add `--single` flag as opt-out for single-GPU mode |

---

## Final File Change Summary

| File | Status | Summary |
|------|--------|---------|
| `scripts/submit_signor_batch.sh` | Modify | Remove 2-job vLLM logic; generate RUN_TAG; **4-chunk parallel is default**; `--single` opt-out; auto-merge dependency job |
| `scripts/slurm_signor_job.sh` | New | Self-contained: start vLLM bg → wait → eval → cleanup; handles array tasks |
| `scripts/run_signor_batch.sh` | Modify | Remove `.vllm_server_info` parsing; use `LLM_BASE_URL` env var directly |
| `experiments/run_signor_eval.py` | Modify | Add `--run-tag` and `--row-start` arguments |

The existing `start_vllm_ihpc.sh` and `vllm_node_setup.sh` remain **unchanged** — they are still useful for interactive/development use.

---

## Resolved Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Merge strategy | **Auto-merge**: dependency job fires automatically after all 4 array tasks succeed |
| 2 | Model path | No change — `vllm_node_setup.sh` already resolves `qwen3.5-9b` → `/hps/nobackup/saezrodriguez/hf_models/qwen3.5-9b` |
| 3 | Default mode | **4-chunk parallel is default**; use `--single` to opt out to single-GPU mode |
