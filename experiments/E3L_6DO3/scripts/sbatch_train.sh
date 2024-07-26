#!/bin/bash
#SBATCH -p gpu-bf
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --time 5:00:00
#SBATCH --job-name=train
#SBATCH -o output.train.log

iter=$1
source ~/.bashrc
conda activate openvs
echo python -u train.py $iter False
python -u train.py $iter False

