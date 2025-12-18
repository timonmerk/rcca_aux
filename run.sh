#!/bin/sh
#SBATCH --mem=20GB
#SBATCH --partition=commons
#SBATCH --time=04:00:00
#SBATCH -o logs/ind_%A_%a.out
#SBATCH -e logs/ind_%A_%a.err
#SBATCH -a 0-3959

OUTDIR="outccn"
uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region all \
    --output_folder $OUTDIR

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region SC \
    --output_folder $OUTDIR

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --region C \
    --output_folder $OUTDIR

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region all \
    --output_folder $OUTDIR

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region SC \
    --output_folder $OUTDIR

uv run rcca_cv.py \
    --idx_run $SLURM_ARRAY_TASK_ID \
    --run-suds \
    --region C \
    --output_folder $OUTDIR


