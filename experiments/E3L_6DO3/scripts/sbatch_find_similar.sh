#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=5g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 12:00:00
#SBATCH --job-name=find_similar_main
#SBATCH -o output.find_similar.log


source ~/.bashrc
conda activate openvs
echo python -u find_similar_zinc22_mol2s.py
python -u find_similar_zinc22_mol2s.py
