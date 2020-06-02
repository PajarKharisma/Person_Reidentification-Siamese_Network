#!/bin/bash
#SBATCH --account=beta-tester     
#SBATCH --partition=TRTest        
#SBATCH --job-name=job            
#SBATCH --ntasks=32               
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2                                                   
#SBATCH --time=20:00:00                           
#SBATCH --output=result-%j.out    
#SBATCH --error=result-%j.err

module load cuda/10.2
module load tensorrt/6-cuda10.1
source /<dir_untuk_virtualenv_tf>/bin/activate

python3 src/main/Main.py
