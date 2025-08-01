#!/bin/tcsh
#SBATCH --job-name=catboost_mpi
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=24
#SBATCH --time=48:00:00
#SBATCH --output=/sciclone/home/tdfelton/baseball/logs/gridsearch_%j.out
#SBATCH --error=/sciclone/home/tdfelton/baseball/logs/gridsearch_%j.err

source /sciclone/home/tdfelton/.tcshrc
conda activate baseball2

cd /sciclone/home/tdfelton/baseball
mpiexec -n 240 python3 /sciclone/home/tdfelton/baseball/submit/gridsearch_mpi.py

