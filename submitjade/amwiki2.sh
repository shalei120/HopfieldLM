#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --job-name=AMLM
#SBATCH --gres=gpu:1

module load cuda/9.2

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo $PWD
# run the application
python3 main.py -m hop_energy -b 128  -d wiki2 -g 0 > slurm-AMLM-wiki2-$SLURM_JOB_ID.out
