#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=10g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 1:00:00
#SBATCH --job-name=gen_3dstruct
#SBATCH -o output.gen_3dstruct.log


iter=$1
source ~/.bashrc
conda activate openvs
echo python prepare_smiles.py $iter
python prepare_smiles.py $iter
echo python gen_3dstruct_jobs.py $iter
python -u gen_3dstruct_jobs.py $iter
sh submit_arrayjobs.gen3d.sh
