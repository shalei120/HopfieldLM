#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --job-name=HopfieldLM
#SBATCH --gres=gpu:1

#module load python3
module load cuda/9.2

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo $PWD
# run the applicationmodule
module load python3/anaconda
conda create -n testenv python=3.8.3 -y
source activate testenv
python3 -m pip install -r requirements.txt
python3 main.py -m asso -b 128  -d wiki2 -g 0 > slurm-hopLM-wiki2-$SLURM_JOB_ID.out
