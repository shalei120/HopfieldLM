#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=devel
#SBATCH --job-name=energyLM
#SBATCH --gres=gpu:1

module load cuda/9.2

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo $PWD
# run the application
python3 main.py -m energy -b 128  -d wiki2 -g 0 -c AM > slurm-energyLM-wiki2-$SLURM_JOB_ID.out
