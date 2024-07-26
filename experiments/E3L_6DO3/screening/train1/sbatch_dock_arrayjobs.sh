#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --time 5:30:00
#SBATCH --job-name=VSX_E3L_6DO3
#SBATCH -o output.VSX_E3L_6DO3.log

source ~/.bashrc
CMD=$(head -$SLURM_ARRAY_TASK_ID E3L_6DO3_dock.joblist | tail -1)
exec ${CMD}
