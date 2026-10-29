#!/usr/bin/env bash
# Run every sweep in configs/ with N local workers (default 8). Each worker takes a disjoint set of
# (dataset, partition, seed) groups from every YAML, so the quantum clean model is trained once per group.
# Runs already in the output CSV are skipped, so the script can be stopped and restarted.
#   scripts/run_validation.sh [workers] [config glob]
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS=${1:-8}
GLOB=${2:-configs/*.yaml}
mkdir -p results/logs
if [[ -d ../.venv ]]; then source ../.venv/bin/activate; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for ((i = 0; i < WORKERS; i++)); do
  (
    for config in $GLOB; do
      python scripts/run_sweep.py --config "$config" --shard "$i" --num-shards "$WORKERS" --skip-existing
    done
    echo "worker $i done $(date -Is)"
  ) > "results/logs/worker_$i.log" 2>&1 &
done
echo "$WORKERS workers started $(date -Is); logs in results/logs/, progress: python scripts/validation_progress.py"
wait
echo "all workers finished $(date -Is)"
