#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=3g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 6:00:00
#SBATCH --job-name=aug_results_main
#SBATCH -o output.aug_results.log

iter=$1
source ~/.bashrc
conda activate openvs
echo python -u augment_vs_results.py $iter
python -u augment_vs_results.py $iter
