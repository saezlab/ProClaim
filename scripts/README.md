# Slurm Runbook

This directory contains the operational scripts for running ConnectomeDB evaluations on EBI HPC with Slurm.

## Main scripts

- `submit_connectomedb_batch.sh`: submit, monitor, cancel, and resume ConnectomeDB Slurm runs from a login machine.
- `slurm_connectomedb_job.sh`: per-task Slurm job entrypoint. Starts a local vLLM server, waits for readiness, then runs the evaluation chunk.
- `run_connectomedb_batch.sh`: local wrapper that invokes the Python batch runner.
- `vllm_node_setup.sh`: launches vLLM on the allocated GPU.
- `setup_ner_venv310.sh`: creates the dedicated `.venv310` environment used by the hard-coded scispaCy NER worker and related Python 3.10 NLP tooling.
- `merge_connectomedb_outputs.py`: merges chunk CSVs and produces a run-level timing summary.
- `summarize_connectomedb_run.py`: local helper for summarizing timing artifacts in a results directory.

## Main CUDA environment

The verifier's `check_sufficiency()` path imports PyTorch from the main project
`.venv`, not from `.venv310` and not from the vLLM container. If that import
fails with missing `libnccl.so.2` or `libcudnn.so.9`, repair the main
environment from the repository root with:

```bash
uv sync --reinstall-package torch \
  --reinstall-package nvidia-cudnn-cu12 \
  --reinstall-package nvidia-nccl-cu12
```

The main `.venv` is provisioned directly from `pyproject.toml`. On Linux
`x86_64`, `uv sync` installs the cu124 PyTorch wheel plus the NCCL/cuDNN runtime
packages required by the sufficiency checker.

## Biomedical NER environment

The feature extraction pipeline launches `scripts/ner_worker.py` through the
hard-coded interpreter path `.venv310/bin/python`. If that environment is
missing, entity coverage is skipped.

Create the expected environment from the repository root with:

```bash
bash scripts/setup_ner_venv310.sh
```

The helper installs the Python 3.10 NLP stack used by the NER worker:

- Python 3.10
- CUDA PyTorch (default: cu124, override with `TORCH_CUDA_INDEX`)
- `numpy<2`
- `spacy>=3.7.4,<3.8`
- `en_core_sci_sm`

This is intentionally separate from the main project `.venv`, which now hosts
the semantic-similarity, NLI, and sufficiency-classifier PyTorch stack.

## Project root resolution

The submit script derives the remote project root from the selected HPC username:

- `--user USER` resolves to `/hps/nobackup/saezrodriguez/USER/workspace/grn-llm-correct`

If the checkout lives elsewhere, override it explicitly:

```bash
GRN_LLM_CORRECT_PROJECT_ROOT=/custom/path/grn-llm-correct \
  bash scripts/submit_connectomedb_batch.sh --user ail --status
```

## Authentication behavior

`submit_connectomedb_batch.sh` now uses SSH connection multiplexing. In the normal case, one run of the script should prompt for the SSH password once, then reuse the same connection for the later `ssh` calls in that invocation.

## Default parallel mode

The default submission mode is an 8-task Slurm array:

- `--array=0-7`
- 1 A100 GPU per task
- one local vLLM server per task
- one ConnectomeDB row chunk per task
- one merge job after the array completes

Submit the default run:

```bash
bash scripts/submit_connectomedb_batch.sh --user ail
```

## Scheduling flow

The batch pipeline has two levels of scheduling:

- Slurm schedules one job in single mode, or an 8-task array in parallel mode.
- Inside each allocated GPU task, the Python evaluator schedules `--workers` concurrent claim jobs against one local vLLM server.

```text
Local machine
  |
  | bash scripts/submit_connectomedb_batch.sh [options]
  v
submit_connectomedb_batch.sh
  |
  |-- parses flags:
  |     --single / default parallel
  |     --reps N
  |     --workers N
  |     --worker-shards
  |     --max-num-seqs N
  |     --limit N
  |
  |-- opens SSH connection to EBI HPC
  |
  |-- submits main Slurm job via sbatch
  |     parallel mode: --array=0-7
  |     single mode:   one job
  |
  |-- exports env into sbatch job:
  |     RUN_TAG
  |     EXTRA_ARGS
  |     SINGLE_MODE=true/false
  |     VLLM_MAX_NUM_SEQS
  |     GRN_LLM_CORRECT_PROJECT_ROOT
  |
  |-- if parallel mode:
  |     submits merge job with dependency afterok:<ARRAY_JOB_ID>
  |
  `-- writes .connectomedb_job_info

HPC / Slurm
  |
  |-- main job starts scripts/slurm_connectomedb_job.sh
  |     one invocation per array task
  |
  |-- array task routing:
  |     task 0 -> rows 0..39
  |     task 1 -> rows 40..79
  |     task 2 -> rows 80..119
  |     task 3 -> rows 120..159
  |     task 4 -> rows 160..199
  |     task 5 -> rows 200..239
  |     task 6 -> rows 240..278
  |     task 7 -> rows 279..317
  |     single mode -> no chunking
  |
  |-- inside each task:
  |     start scripts/vllm_node_setup.sh in background
  |     wait for port file
  |     poll localhost health endpoint
  |     export LLM_BASE_URL=http://localhost:<port>/v1/
  |     run scripts/run_connectomedb_batch.sh
  |
  `-- on exit: kill local vLLM and clean temp file

Per Slurm task
  |
  |-- vllm_node_setup.sh
  |     detects visible GPU count
  |     finds free port
  |     starts:
  |       vllm serve MODEL
  |         --tensor-parallel-size <gpu_count>
  |         --port <port>
  |         --max-num-seqs <VLLM_MAX_NUM_SEQS>
  |
  `-- result: one local OpenAI-compatible vLLM endpoint

Python evaluator
  |
  |-- run_connectomedb_batch.sh
  |     uv run python experiments/run_connectomedb_eval.py ...
  |
  |-- run_connectomedb_eval.py
  |     reads dataset
  |     applies --row-start/--limit from the Slurm wrapper
  |     expands each row into claim repetitions
  |     skips already completed runs
  |     pushes pending jobs into a queue
  |
  |-- starts --workers N threads
  |     each thread:
  |       pop next (row, rep) job
  |       run verifier subprocess
  |       subprocess uses LLM_BASE_URL -> local vLLM
  |       append result to task CSV or worker shard CSV
  |
  `-- if --worker-shards:
        merge per-worker CSVs into the task CSV

Parallel mode finalization
  |
  `-- merge job runs scripts/merge_connectomedb_outputs.py
        results_chunk0.csv + ... + results_chunk7.csv
        -> results.csv
```

Practical interpretation:

- `--workers N` increases client-side concurrency inside one Slurm task.
- `--max-num-seqs N` increases server-side concurrency on that task's vLLM instance.
- In parallel mode, total throughput is the combination of Slurm array fan-out across GPUs and worker fan-out within each GPU task.

## Single-GPU smoke test

For smoke tests, request less wall time. Long wall-time requests can wait longer in Slurm because they are harder to backfill.

Recommended first smoke test:

```bash
bash scripts/submit_connectomedb_batch.sh \
  --user ail \
  --single \
  --limit 2 \
  --reps 1 \
  --workers 2 \
  --worker-shards \
  --max-num-seqs 16 \
  --time 02:00:00
```

What this does:

- uses one GPU instead of the 8-task array
- evaluates only 2 claims
- runs 2 concurrent claim workers against one local vLLM server
- writes one shard CSV per worker and merges them into the task CSV
- asks Slurm for 2 hours instead of 40

If queue pressure is high, try `--time 01:00:00` only if you expect startup plus evaluation to fit comfortably.

## Packed-mode flags

The submission script exposes the main packed-mode controls:

- `--workers N`: number of concurrent claim repetitions per GPU task
- `--worker-shards`: write one result CSV per worker, then merge them into the task CSV
- `--max-num-seqs N`: vLLM concurrency per GPU task
- `--time HH:MM:SS`: Slurm wall-time request

Example packed single-task run:

```bash
bash scripts/submit_connectomedb_batch.sh \
  --user ail \
  --single \
  --reps 1 \
  --workers 2 \
  --worker-shards \
  --max-num-seqs 16 \
  --time 04:00:00
```

## Monitoring and control

Check queue state and result progress:

```bash
bash scripts/submit_connectomedb_batch.sh --user ail --status
```

Follow Slurm logs:

```bash
bash scripts/submit_connectomedb_batch.sh --user ail --logs
```

Cancel the active run tracked in `.connectomedb_job_info`:

```bash
bash scripts/submit_connectomedb_batch.sh --user ail --cancel
```

Resume the last tracked run tag:

```bash
bash scripts/submit_connectomedb_batch.sh --user ail --resume
```

Resume an explicit run tag:

```bash
bash scripts/submit_connectomedb_batch.sh --user ail --run-tag 20260424_171117 --reps 1
```

## Outputs

Each run writes into:

- `results/connectomedb_eval_<RUN_TAG>/`

Typical outputs:

- `results_chunk0.csv` ... `results_chunk7.csv`: per-task result CSVs in array mode
- `results.csv`: merged final CSV
- `results_chunk*_timing.json`: per-task timing reports
- `timing_summary.json`: merged timing summary written by `merge_connectomedb_outputs.py`
- `results_chunk*.worker*.csv`: per-worker shard CSVs when `--worker-shards` is enabled

## Local post-run summary

To summarize an existing results directory locally:

```bash
python scripts/merge_connectomedb_outputs.py \
  --results-dir results/connectomedb_eval_<RUN_TAG> \
  --skip-merge
```

Or use the standalone timing summarizer:

```bash
python scripts/summarize_connectomedb_run.py \
  --results-dir results/connectomedb_eval_<RUN_TAG>
```

## Operational notes

- The compute-side job starts vLLM first and only then launches the evaluation.
- `LLM_BASE_URL` is set automatically inside the Slurm task.
- For packed runs, start with `--workers 2 --max-num-seqs 16` before pushing concurrency further.
- If jobs stay pending for too long, reducing `--time` is often the first useful scheduling adjustment for smoke tests.
- If the remote checkout path does not match the derived username path, set `GRN_LLM_CORRECT_PROJECT_ROOT` explicitly.
