#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=devel
#SBATCH --job-name=TransformerLM
#SBATCH --gres=gpu:1

module load cuda/9.2

#echo $CUDA_VISIBLE_DEVICES
#nvidia-smi
echo $PWD
# run the application
python3 main.py -m transformer -b 128  > slurm-transformerLM-$SLURM_JOB_ID.out
