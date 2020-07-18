#!/bin/bash
#SBATCH --account=cbt
#SBATCH --partition=zentwo
#SBATCH --job-name=job
#SBATCH --ntasks=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00
#SBATCH --output=log/result/result-%j.out
#SBATCH --error=log/result/result-%j.err

module load anaconda3
eval “$(conda shell.bash hook)”
conda activate $WORK/.venv

#%Module
module load cuda/10.1-cuDNN7.6.5
module load tensorrt/6-cuda10.1

python3 src/main/Main.py
