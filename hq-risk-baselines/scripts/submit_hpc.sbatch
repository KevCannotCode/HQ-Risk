#!/bin/bash
#SBATCH --job-name=hq-risk-baselines
#SBATCH --array=0-39
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/%x_%A_%a.out
#
# Runs every sweep, split into $SLURM_ARRAY_TASK_COUNT shards, one shard per array task.
# Each shard writes its own CSV so no two tasks touch the same file on a shared filesystem.
# Afterwards, from the login node:  python scripts/merge_shards.py && python scripts/build_summary.py
#
# Cluster name, partition, module loads and account are unknown (see METHODS.md, blocker table).
# Uncomment and fill once known:
##SBATCH --partition=<cluster-partition>
##SBATCH --account=<account>
# module load python/3.12

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/raw/shards results/logs

if [[ -d .venv ]]; then source .venv/bin/activate; fi

for config in configs/*.yaml; do
    name=$(basename "$config" .yaml)
    python scripts/run_sweep.py \
        --config "$config" \
        --shard "$SLURM_ARRAY_TASK_ID" \
        --num-shards "$SLURM_ARRAY_TASK_COUNT" \
        --output "results/raw/shards/${name}_${SLURM_ARRAY_TASK_ID}.csv"
done
