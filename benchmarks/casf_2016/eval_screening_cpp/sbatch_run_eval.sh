#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=30g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 2-00:00:00
#SBATCH --job-name=eval_main
#SBATCH -o output.eval_main.log


source ~/.bashrc
conda activate deepdock
python -u run_eval.py
