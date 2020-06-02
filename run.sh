#!/bin/bash
#SBATCH --account=beta-tester     
#SBATCH --partition=TRTest        
#SBATCH --job-name=job            
#SBATCH --ntasks=32               
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2                                                   
#SBATCH --time=20:00:00                           
#SBATCH --output=log/result/result-%j.out    
#SBATCH --error=log/result/result-%j.err

#%Module
module load tensorrt/6-cuda10.2
source .venv/bin/activate

python3 src/main/Main.py
