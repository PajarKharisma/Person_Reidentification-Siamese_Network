#!/bin/bash
#
#SBATCH --job-name=job
#SBATCH --output=log/result/result-%j.out
#SBATCH --error=log/result/result-%j.err
#
#SBATCH --nodes=1
#SBATCH --time=20:00:00

source ../.venv/bin/activate
srun python3 src/main/Main.py