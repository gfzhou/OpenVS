#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=3g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 6:00:00
#SBATCH --job-name=gather_vs_results
#SBATCH -o output.gather_vs_results.log

iter=$1
source ~/.bashrc
conda activate openvs
echo python -u gather_vs_results.py $iter
python gather_vs_results.py $iter
