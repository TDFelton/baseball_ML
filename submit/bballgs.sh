#!/bin/bash
#SBATCH --job-name=catboost_grid
#SBATCH --output=/sciclone/home/tdfelton/baseball/logs/gridsearch_%j.out
#SBATCH --error=/sciclone/home/tdfelton/baseball/logs/gridsearch_%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tdfelton@wm.edu

# Load environment
pwd; hostname; date
source ~/.bashrc
conda activate baseball2

# Set temporary directory if needed
export TMPDIR=/local/tmp/$USER/$SLURM_JOBID
mkdir -p $TMPDIR

# Run your job
python3 /sciclone/home/tdfelton/baseball/two_stage.py

date
