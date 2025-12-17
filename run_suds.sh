#!/bin/sh
#SBATCH --mem=20GB
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH -o logs/ind_%A_%a.out
#SBATCH -e logs/ind_%A_%a.err
#SBATCH -a 0-1975


uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region all \
    --output_folder outl

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region SC \
    --output_folder outl

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region C \
    --output_folder outl

