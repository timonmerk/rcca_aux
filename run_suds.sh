#!/bin/sh
#SBATCH --mem=20GB
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH -o logs/ind_%A_%a.out
#SBATCH -e logs/ind_%A_%a.err
#SBATCH -a 0-1567


uv run rcca_cv.py \
    $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region all \
    --output_folder out

uv run rcca_cv.py \
    $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region SC \
    --output_folder out

uv run rcca_cv.py \
    $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region C \
    --output_folder out

