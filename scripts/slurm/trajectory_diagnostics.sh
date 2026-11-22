#!/bin/bash
#SBATCH --account=rockhpc_mcposd
#SBATCH --job-name=qg_trajectory_diagnostics
#SBATCH --output=logs/trajectory-diagnostics-%j.out
#SBATCH --error=logs/trajectory-diagnostics-%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=1:00:00

set -euo pipefail

source /mnt/nfs/home/c5044892/repos/qg_project/miniconda/etc/profile.d/conda.sh
conda activate QG

cd "${SLURM_SUBMIT_DIR}"

DATA_DIR="${DATA_DIR:-data/data_hr512_nx32_13}"
CONFIG="${CONFIG:-config/default.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/${USER}/qg_trajectory_diagnostics_${SLURM_JOB_ID}}"

mkdir -p logs "${OUTPUT_DIR}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun python scripts/trajectory_diagnostics.py \
    --config "${CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}"
