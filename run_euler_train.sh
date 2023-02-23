#!/bin/bash

#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --output /cluster/home/tmehmet/ms-convSTAR-swiss-map/euluer_outputs/outfile_train2021_%J.%I.txt
#SBATCH --mem-per-cpu=80000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --mail-type=END,FAIL
# ## ##SBATCH -R "rusage[scratch=80000]"


# Evaluation
python3 train_ensemble.py -w 1 -b 16 --hidden 128 --seed 1 -exp 50601