#!/bin/sh
#SBATCH --mem=20GB
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH -o logs/ind_%A_%a.out
#SBATCH -e logs/ind_%A_%a.err
#SBATCH -a 0-4559 # 3167

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region all \
    --output_folder outl

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region SC \
    --output_folder outl

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region C \
    --output_folder outl
