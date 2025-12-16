#!/bin/sh
#SBATCH --mem=20GB
#SBATCH --partition=commons
#SBATCH --time=02:00:00
#SBATCH -o logs/ind_%A_%a.out
#SBATCH -e logs/ind_%A_%a.err
#SBATCH -a 0-35 # 19007

uv run rcca_run.py \
    $SLURM_ARRAY_TASK_ID \
    --run_suds \
    --region all \
    --output_folder outercv

uv run rcca_run.py \
    $SLURM_ARRAY_TASK_ID \
    --run_suds \
    --region SC \
    --output_folder outercv

uv run rcca_run.py \
    $SLURM_ARRAY_TASK_ID \
    --run_suds \
    --region C \
    --output_folder outercv

