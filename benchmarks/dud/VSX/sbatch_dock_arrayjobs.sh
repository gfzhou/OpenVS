#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --time 5:30:00
#SBATCH --job-name=vsx_dud
#SBATCH -o output.vsx_dud.log

source ~/.bashrc
CMD=$(head -$SLURM_ARRAY_TASK_ID dud_dock.joblist | tail -1)
exec ${CMD}
