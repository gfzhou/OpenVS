#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=30g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 24:00:00
#SBATCH --job-name=save_top_main
#SBATCH -o output.save_top_main.log

iter=$1
source ~/.bashrc
conda activate openvs
echo python -u save_top_predictions.py $iter
python -u save_top_predictions.py $iter
