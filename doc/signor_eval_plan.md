# SIGNOR End-to-End Evaluation Implementation Plan

## 1. Objective
Create a unified Python evaluation script (`experiments/run_signor_eval.py`) to systematically run the RLM evidence verification pipeline (`evidence_programming.py`) across the entire SIGNOR structural ground truth dataset.

## 2. Input Data
- **File**: `data/signor/ground_truth.csv`
- **Total Claims**: 67 rows (excluding header).
- **Relevant Columns**: `ENTITYA`, `ENTITYB`, `EFFECT`, `MECHANISM`, `Label` (Ground Truth), `SIGNOR_ID`

## 3. Core Logic (Finalized)

### A. Claim Construction
Based on `src/pkevolve/utils/signor_utils.py`, we will use a `construct_signor_claim(source, target, interaction)` function that removes "Does" from the question to form a definitive claim:
- **Positive Interactions** (e.g., up-regulates, up-regulates activity):
   `"{source} directly activates {target} (either through activation or increase of expression)."`
- **Negative Interactions** (e.g., down-regulates, down-regulates activity):
   `"{source} directly inhibits {target} (either through inhibition or destabilization)."`
- **Other Interactions** (e.g., unknown, complex):
   `"{source} physically interacts with {target}."`

### B. "Flip Flag" Definition
Flipping a claim inverses the **interaction** (not the entities) and maps the **Ground Truth Label** appropriately:
- **Positive \<\-\> Negative**: "directly activates" \<-\> "directly inhibits"
- **Physical \<\-\> No Physical**: "physically interacts with" \<-\> "does not physically interact with"
- **Label Mapping**: 
  - `SUPPORTED` \<-\> `WRONG` (or `REFUTED` depending on dataset terms)
  - `UNCERTAIN` \<-\> `UNCERTAIN`

### C. Repetitions & Scale
- Each row (Forward and Flipped) evaluates **3 times**.
- 67 rows * 2 variations * 3 reps = **402 total executions**.

### D. Token Usage and Cost
The script will capture token usage from the log output (or the agent `ResultMessage`) and estimate the cost for each run, appending this to the output CSV.

## 4. Proposed Script Architecture (`experiments/run_signor_eval.py`)

1. **Argument Parsing:** Accept paths for the input CSV (`/hps/nobackup/saezrodriguez/shared_datasets/signor*/ground_truth.csv`), output CSV, target config (`experiments/example_config.yaml`), and number of repetitions.
2. **Resumable Execution:** 
   - The script will use isolated output directories for every single run:
     `results/signor_eval/{SIGNOR_ID}/flip_{True|False}/rep_{1|2|3}/`
   - Before launching the Claude Agent SDK for a run, the script will check if `verdict.json` already exists in the target directory. If it does, it **skips** the run. This allows the script to be stopped and safely resumed at any time.
3. **Subprocess Isolation:** 
   - Each claim evaluation will be invoked using `subprocess.run(["uv", "run", "python", "-m", "pkevolve.verification.evidence_programming", ...])`.
   - This ensures memory and asyncio loops are perfectly isolated between runs.
4. **Result Aggregation:** 
   - After each run, the script parses the localized `verdict.json`.
   - It appends a new row to `results/signor_eval_results.csv`.
   - **Output CSV Columns**: `SIGNOR_ID`, `ENTITYA`, `ENTITYB`, `Original_Label`, `Flipped_Label`, `Claim_String`, `Is_Flipped`, `Repetition`, `Agent_Verdict`, `Agent_Confidence`, `Reasoning_Snippet`, `Output_Directory`, `Input_Tokens`, `Output_Tokens`, `Cost_Estimate`.

## 5. Usage / How to Run

To run the full evaluation pipeline, you need to first start the vLLM subagent server on the HPC, and then run the local batch orchestrator:

### Step 1: Start the vLLM Server (HPC)
Open a terminal and run the provided SLURM script. This will request an HPC node, load the model, and establish an SSH tunnel to your local machine at `localhost:8000`:
```bash
bash scripts/start_vllm_ihpc.sh
```
*(Wait until the terminal shows "vLLM server is READY!")*

### Step 2: Run the Evaluation (Local)
In a new terminal (or inside a local `tmux` session), execute the local runner script. It automatically configures `LLM_BASE_URL` to point to your SSH tunnel and begins testing claims sequentially:

**Run the entire dataset (402 runs, resumable):**
```bash
scripts/run_signor_batch.sh
```

**Test a single claim (useful for debugging):**
```bash
scripts/run_signor_batch.sh --limit 1
```

**Run with fewer repetitions (e.g., 1 pass instead of 3):**
```bash
scripts/run_signor_batch.sh --reps 1
```

### Monitoring Progress
- The script outputs its progress directly to the terminal.
- Structured results (Verdicts, Confidence, Token count, Costs) are appended line-by-line to `results/signor_eval_results.csv`.
- Detailed execution traces and Jupyter evidence notebooks are isolated and saved under:
  `results/signor_eval/<SIGNOR_ID>/flip_<True|False>/rep_<1|2|3>/`
