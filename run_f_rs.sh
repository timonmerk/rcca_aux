#!/bin/sh
#SBATCH --mem=20GB
#SBATCH --partition=commons
#SBATCH --time=02:00:00
#SBATCH -o logs/ind_%A_%a.out
#SBATCH -e logs/ind_%A_%a.err
#SBATCH -a 0-35 # 19007

OUTDIR="outercvl"

uv run rcca_run.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region all \
    --output_folder $OUTDIR

uv run rcca_run.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region SC \
    --output_folder $OUTDIR

uv run rcca_run.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region C \
    --output_folder $OUTDIR


