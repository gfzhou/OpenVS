#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=5g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 10:00:00
#SBATCH --job-name=extract_mol2
#SBATCH -o output.extract_mol2.log


iter=$1
source ~/.bashrc
conda activate openvs
echo python -u extract_zinc22_raw_mol2s.py $iter
python -u extract_zinc22_raw_mol2s.py $iter
