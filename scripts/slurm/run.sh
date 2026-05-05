#!/bin/bash
#SBATCH --account=rockhpc_mcposd
#SBATCH --job-name=qg_cpu_array
#SBATCH --output=logs/slurm-%A.out
#SBATCH --error=logs/slurm-%A.err
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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

