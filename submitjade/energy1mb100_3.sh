#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --job-name=energyLM
#SBATCH --gres=gpu:1

module load cuda/9.2

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo $PWD
# run the application
python3 main.py -m energy -b 64  -d 1mb -g 0 -emb 100 -layer 3 > slurm-energyLM-1mb-$SLURM_JOB_ID-100-3.out
