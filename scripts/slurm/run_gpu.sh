#!/bin/bash
#SBATCH --account=rockhpc_mcposd
#SBATCH --job-name=2048to128
#SBATCH --output=logs/slurm-%A.out
#SBATCH --error=logs/slurm-%A.err
#SBATCH --partition=gpu-s_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

source /mnt/nfs/home/c5044892/repos/qg_project/miniconda/etc/profile.d/conda.sh
conda activate QG

# Change to repository root
cd $SLURM_SUBMIT_DIR

# Run 
OUTDIR=/scratch/$USER/qg_out_${SLURM_JOB_ID}
mkdir -p $OUTDIR
python run.py --config config/default.yaml --outdir $OUTDIR

