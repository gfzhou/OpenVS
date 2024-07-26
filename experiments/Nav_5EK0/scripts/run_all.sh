#!/bin/bash
# wrapper for sbatch
submit() {
    if [ -n "$prev_jobid" ]; then
        job_id=$(sbatch --dependency=afterok:$prev_jobid "$@" | sed 's/Submitted batch job //')
    else
        job_id=$(sbatch "$@" | sed 's/Submitted batch job //')
    fi
    echo $job_id # the "return" value
    echo $job_id >>/dev/stderr
}

# make the script stop when error (non-true exit code) is occured
set -e

curr_iter=1
#prev_jobid=$(submit sbatch_gather_vs_results.sh $curr_iter)
#echo sbatch_gather_vs_results.sh $prev_jobid

#prev_jobid=$(submit sbatch_augment_results.sh $curr_iter)
#echo sbatch_augment_results.sh $prev_jobid

#prev_jobid=$(submit -o output.train.$curr_iter.log sbatch_train.sh $curr_iter)
#echo sbatch_train.sh $prev_jobid

#prev_jobid=$(submit sbatch_predict_slurm.sh $curr_iter)
#echo sbatch_predict_slurm.sh $prev_jobid

#prev_jobid=$(submit sbatch_save_top_prediction.sh $curr_iter)
#echo sbatch_save_top_prediction.sh $prev_jobid

#prev_jobid=$(submit sbatch_extract_mol2s.sh $curr_iter)
#echo submit sbatch_extract_mol2s.sh $prev_jobid

#prev_jobid=$(submit sbatch_gen_params.sh $(($curr_iter+1)))
#echo sbatch_gen_params.sh $prev_jobid

