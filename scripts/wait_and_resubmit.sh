#!/usr/bin/env bash
#SBATCH --job-name=wait-resubmit
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct/results/slurm_logs/wait-resubmit-%j.out
#SBATCH --error=/hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct/results/slurm_logs/wait-resubmit-%j.err
# Wait for SLURM job array to finish, then switch mlp_model_dir and resubmit.
# Edit JOB_ID and the sed substitution below before submitting.
# sbatch scripts/wait_and_resubmit.sh
set -euo pipefail

JOB_ID=19086598
CONFIG="experiments/configs/signor_direct_config.yaml"
POLL_INTERVAL=600   # check every 10 minutes
MAX_WAIT=36000      # give up after 10 hours

cd /hps/nobackup/saezrodriguez/rain/workspace/grn-llm-correct

echo "[$(date)] Watching job ${JOB_ID} — polling every $((POLL_INTERVAL/60))m, timeout 10h"

elapsed=0
while [[ $elapsed -lt $MAX_WAIT ]]; do
    RUNNING=$(squeue --job "${JOB_ID}" --noheader 2>/dev/null | wc -l)
    if [[ "$RUNNING" -eq 0 ]]; then
        echo "[$(date)] Job ${JOB_ID} no longer in queue."
        break
    fi
    echo "[$(date)] Still running (${RUNNING} task(s) active). Next check in $((POLL_INTERVAL/60))m..."
    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [[ $elapsed -ge $MAX_WAIT ]]; then
    echo "[$(date)] ERROR: timed out after 10h — job ${JOB_ID} still running. Exiting without resubmit."
    exit 1
fi

# Swap mlp_model_dir from tau_0.00_seed_42 → tau_0.25_seed_42
sed -i 's|^mlp_model_dir: results/ablation/models/tau_0.00_seed_42$|mlp_model_dir: results/ablation/models/tau_0.25_seed_42|' "${CONFIG}"
echo "[$(date)] Config updated:"
grep "mlp_model_dir" "${CONFIG}"

echo "[$(date)] Submitting next batch..."
bash scripts/submit_signor_direct_batch.sh --reps 1
echo "[$(date)] Submission done."
