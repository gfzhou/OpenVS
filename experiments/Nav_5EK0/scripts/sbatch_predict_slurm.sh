#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=predict_db_main
#SBATCH -o output.predict_db_slurm.log

iter=$1
source ~/.bashrc
conda activate openvs
echo python -u predict_db.py --i_iter $iter --run_platform 'slurm'
python -u predict_db.py --i_iter $iter --run_platform 'slurm'
