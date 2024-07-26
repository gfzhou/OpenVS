#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=10g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 12:00:00
#SBATCH --job-name=gen_params_main
#SBATCH -o output.gen_params.log


source ~/.bashrc
conda activate openvs
iter=$1
echo python -u gen_params.py $iter
python -u gen_params.py $iter
