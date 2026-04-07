#!/bin/bash
#SBATCH --account=bdncl19
#SBATCH --job-name=cnn_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

# Load modules - adjust to your cluster's modules
module purge
module load Miniforge
module load cuda

conda activate base

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to repository root
cd $SLURM_SUBMIT_DIR

# Run 
OUTDIR=/scratch/$USER/qg_out_${SLURM_JOB_ID}
mkdir -p $OUTDIR
python run.py --config config/default.yaml --outdir $OUTDIR
