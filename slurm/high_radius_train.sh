#!/bin/bash
#SBATCH --job-name=hr_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=itc_gpu
#SBATCH --output=/n/home03/hbrittain/slurm_jobs/out_files/high_radius_train-%j.out
#SBATCH --error=/n/home03/hbrittain/slurm_jobs/out_files/high_radius_train-%j.err

set -euo pipefail

mkdir -p /n/home03/hbrittain/slurm_jobs/out_files
cd /n/home03/hbrittain/halognn/networks/DeepHalos
export PYTHONPATH="$PWD:$PYTHONPATH"
export HALOGNN_HIGH_RADIUS_MODE=train
export MPLBACKEND=Agg

source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate ourenv

python -u scripts/high_radius_sweep.py
