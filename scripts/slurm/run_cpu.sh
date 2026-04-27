#!/bin/bash
#SBATCH --account=rockhpc_mcposd
#SBATCH --job-name=qg_cpu_array
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Total combinations for the suggested CNN sweep in config/default.yaml: 3*3*3*2 = 54
# Adjust the array range to match number of combinations (0..53). Throttle 10 concurrent jobs.
#SBATCH --array=0-53%10

source /mnt/nfs/home/c5044892/repos/qg_project/miniconda/etc/profile.d/conda.sh
conda activate QG

CONFIG="config/default.yaml"

# SLURM_ARRAY_TASK_ID provides the index of this task
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Optional per-task outdir to avoid clashes
OUTDIR="outputs/cpu_array_job_${SLURM_JOB_ID}_task_${TASK_ID}"

# Ensure logs/out exists
mkdir -p logs

# Compute a short hash of the config to create a unique marker for this
# sweep's dataset. Use a lock+marker pattern so only one task generates data.
CONFIG_HASH=$(python - <<PY
import hashlib
print(hashlib.sha1(open('${CONFIG}','rb').read()).hexdigest())
PY
)

DATA_DIR="data"
mkdir -p "${DATA_DIR}"
DATAFLAG="${DATA_DIR}/.data_ready_${CONFIG_HASH}"
LOCKDIR="${DATA_DIR}/.data_lock_${CONFIG_HASH}"

if [ -f "${DATAFLAG}" ]; then
	echo "Data ready marker found: ${DATAFLAG}"
else
	if mkdir "${LOCKDIR}" 2>/dev/null; then
		echo "Acquired data-generation lock (${LOCKDIR}), generating data..."
		python run.py --config ${CONFIG} --generate-only
		rc=$?
		if [ $rc -ne 0 ]; then
			echo "Data generation failed (rc=$rc); releasing lock and exiting"
			rmdir "${LOCKDIR}"
			exit $rc
		fi
		touch "${DATAFLAG}"
		rmdir "${LOCKDIR}"
	else
		echo "Another task is generating data; waiting for ${DATAFLAG}"
		while [ ! -f "${DATAFLAG}" ]; do
			sleep 10
		done
	fi
fi

# Run the helper which will pick the right combo and call run.py
srun python scripts/run_cpu.py --config ${CONFIG} --task-index ${TASK_ID} --outdir-override ${OUTDIR}
