#!/bin/bash

#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=20000,ngpus_excl_p=2]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -J "swiss_map"
#BSUB -R "rusage[scratch=2000]"
#BSUB -o /cluster/home/tmehmet/1.txt

export CUDA_VISIBLE_DEVICES=0,1
printenv CUDA_VISIBLE_DEVICES

python3 film_naive_euler.py 
