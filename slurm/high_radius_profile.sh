#!/bin/bash
#SBATCH --job-name=hr_profile
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-02:00
#SBATCH --partition=test
#SBATCH --output=/n/home03/hbrittain/slurm_jobs/out_files/high_radius_profile-%j.out
#SBATCH --error=/n/home03/hbrittain/slurm_jobs/out_files/high_radius_profile-%j.err

set -euo pipefail

mkdir -p /n/home03/hbrittain/slurm_jobs/out_files
cd /n/home03/hbrittain/halognn/networks/DeepHalos
export PYTHONPATH="$PWD:$PYTHONPATH"
export HALOGNN_HIGH_RADIUS_MODE=profile
export MPLBACKEND=Agg

source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate ourenv

python -u scripts/high_radius_sweep.py
