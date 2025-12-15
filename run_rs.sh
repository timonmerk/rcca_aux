#!/bin/sh
#SBATCH --mem=20GB
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH -o logs/ind_%A_%a.out
#SBATCH -e logs/ind_%A_%a.err
#SBATCH -a 3000-3455 # 3455

uv run try_rcca.py $SLURM_ARRAY_TASK_ID 0