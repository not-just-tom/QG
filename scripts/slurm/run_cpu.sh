#!/bin/bash
## Sample SLURM array script to run `run_cpu.py` across sweep combinations.
## Adjust SBATCH directives below for your cluster (partition, time, mem, cpus-per-task).
## Replace <N-1> in --array with (number_of_combinations - 1).

#SBATCH --job-name=qg_cpu_array
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-<N-1>%10

module purge
module load anaconda/3
source activate base

CONFIG="config/default.yaml"

# SLURM_ARRAY_TASK_ID provides the index of this task
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Optional per-task outdir to avoid clashes
OUTDIR="outputs/cpu_array_job_${SLURM_JOB_ID}_task_${TASK_ID}"

# Ensure logs/out exists
mkdir -p logs

# Run the helper which will pick the right combo and call run.py
srun python run_cpu.py --config ${CONFIG} --task-index ${TASK_ID} --outdir-override ${OUTDIR}
