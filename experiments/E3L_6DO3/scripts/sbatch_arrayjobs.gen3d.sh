#!/bin/bash
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=500m
#SBATCH --time 6:30:00
#SBATCH --job-name=gen3d
#SBATCH -o output.gen3d.log

source ~/.bashrc
conda activate openvs
CMD=$(head -$SLURM_ARRAY_TASK_ID ./gen3d.joblist | tail -1)
exec ${CMD}
