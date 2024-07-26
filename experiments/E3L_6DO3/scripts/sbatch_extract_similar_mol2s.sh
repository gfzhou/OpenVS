#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=10g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 6:00:00
#SBATCH --job-name=extract_similiar
#SBATCH -o output.extract_similiar_mol2.log


source ~/.bashrc
conda activate openvs
echo python -u extract_similiar_zinc22_raw_mol2s.py
python -u extract_similiar_zinc22_raw_mol2s.py
