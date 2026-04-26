#!/usr/bin/env bash
# Submit the array job using the sweep in a config, then submit an aggregator job dependent on the array.
# Usage: ./submit_array_and_aggregate.sh [config.yaml] [concurrency]

CONFIG=${1:-config/default.yaml}
CONCURRENCY=${2:-10}

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

# Compute number of combinations from the chosen model_type in the config
read -r TOTAL <<< $(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
model_type = cfg.get('ml', {}).get('model_type')
archs = cfg.get('architectures', {})
arch_cfg = archs.get(model_type, {}) if archs else {}
lists = [v for v in arch_cfg.values() if isinstance(v, list) and len(v) > 1]
from functools import reduce
import operator
if lists:
    lens = [len(l) for l in lists]
    total = reduce(operator.mul, lens, 1)
else:
    total = 1
print(total)
PY
)

if [ -z "$TOTAL" ]; then
  echo "Failed to compute total combinations from $CONFIG" >&2
  exit 1
fi

if [ "$TOTAL" -le 1 ]; then
  echo "Single-job run (no array). Submitting single job."
  SUB_OUT=$(sbatch scripts/slurm/run_cpu.sh)
else
  LAST_INDEX=$((TOTAL - 1))
  echo "Submitting array job with indices 0..$LAST_INDEX (total $TOTAL)"
  SUB_OUT=$(sbatch --array=0-${LAST_INDEX}%${CONCURRENCY} scripts/slurm/run_cpu.sh)
fi

echo "$SUB_OUT"
ARRAY_ID=$(echo "$SUB_OUT" | awk '{print $4}')
if [ -z "$ARRAY_ID" ]; then
  echo "Failed to parse job id from sbatch output: $SUB_OUT" >&2
  exit 1
fi

echo "Submitted job id: $ARRAY_ID"

# Submit aggregator dependent on array success
echo "Submitting aggregator job dependent on array $ARRAY_ID"
sbatch --dependency=afterok:${ARRAY_ID} --output=logs/agg_%A.out --error=logs/agg_%A.err \
  --job-name=qg_aggregate --time=00:30:00 --ntasks=1 \
  --wrap="source /mnt/nfs/home/c5044892/repos/qg_project/miniconda/etc/profile.d/conda.sh; conda activate QG; python scripts/aggregate_results.py --outputs outputs --top 5 > outputs/aggregate_top5.txt"

echo "Aggregator submitted (afterok:${ARRAY_ID})."
