#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --output /cluster/home/tmehmet/ms-convSTAR-swiss-map/euluer_outputs/outfile_%J.%I.txt
#SBATCH --mem-per-cpu=13000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --mail-type=END,FAIL
# ## ##SBATCH -R "rusage[scratch=240000]"


# Evaluation
python3 train_euler.py --eval -w 1 -b 16 --hidden 128 -exp 9999999