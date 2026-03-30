#!/bin/bash
#SBATCH --job-name=impute_array
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-6
#SBATCH --time=04:00:00
#SBATCH --mem=16GB

module load python

source myenv/bin/activate

python main.py $SLURM_ARRAY_TASK_ID