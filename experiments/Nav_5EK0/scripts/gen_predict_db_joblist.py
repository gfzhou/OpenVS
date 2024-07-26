import os,sys
from glob import glob
import numpy as np
import orjson
from tap import Tap

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

class RunArgs(Tap):
    i_iter: int
    fplist: str

def write_sbatchfn(sbatchfn, submitfn, joblistfn, n_jobs, job_name="arrayjobs", queue='cpu'):
    if queue=='cpu':
        sbatch_content = """#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=5g
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 3:00:00
#SBATCH --job-name={jobname}
#SBATCH -o output.{jobname}.log

source ~/.bashrc
CMD=$(head -$SLURM_ARRAY_TASK_ID {joblistfn} | tail -1)
exec ${{CMD}}
""".format(jobname=job_name, joblistfn=joblistfn)
        with open(sbatchfn, 'w') as outf:
            outf.write(sbatch_content)
        print("Wrote: %s"%sbatchfn)

        submit_content = """#!/bin/bash
sbatch -a 1-{}%50 {}
""".format( n_jobs,os.path.basename(sbatchfn) )
        with open(submitfn, 'w') as outf:
            outf.write(submit_content)
        print("Wrote: %s"%submitfn)

def save_listfns(inlist, outdir, prefix="fpfilelist", batchsize=100):
    os.makedirs(outdir, exist_ok=True)
    startIndex = np.arange(0, len(inlist), batchsize)
    content = []
    for i, startndx in enumerate(startIndex):
        outfn = os.path.join(outdir, f"{prefix}.{i}.txt")
        if os.path.exists(outfn):
            raise Exception(f"{outfn} exists.")
        sublist = inlist[startndx:startndx+batchsize]
        content = [f"{fpfn}\n" for fpfn in sublist]
        if len(content) == 0: continue
        with open(outfn, 'w') as outf:
            outf.writelines(content)
        print(f"Saved: {outfn}")

def gen_joblist(i_iter, listfiledir, joblistfn="predict_db.joblist"):
    
    patt = os.path.join(listfiledir, "*.txt")
    filelist = sorted( glob(patt) )
    app = "predict_db_arraytask.py"
    platform = 'cpu'
    joblist = []
    for infn in filelist:
        cmd = f"{app} --i_iter {i_iter} --fplist {infn} --run_platform {platform}\n"
        joblist.append(cmd)
    
    with open(joblistfn, 'w') as outf:
        outf.writelines(joblist)
    print(f"Saved: {joblistfn}")
    
    sbatchfn="sbatch_predict_db_array.sh"
    submitfn = "submit_predict_db_array.sh"
    n_jobs = len(joblist)
    write_sbatchfn(sbatchfn, submitfn, joblistfn, n_jobs, job_name="predict_db", queue='cpu')

def gen_joblist_zinc22(i_iter, config):
    fps_path = config["fps_path"]
    dbfn_pattern = os.path.join(fps_path, "zinc-22*", "*.feather")
    dbfns = sorted(glob(dbfn_pattern))
    fpfilelist_outpath = "./fpfilelist"
    save_listfns(dbfns, fpfilelist_outpath, prefix="fpfilelist", batchsize=100)
    gen_joblist(i_iter, fpfilelist_outpath, joblistfn="predict_db.joblist")

def main():
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    
    configfn = os.path.join("../", "config_zinc22_db.json" )
    config = load_configfn(configfn)
    gen_joblist_zinc22(i_iter, config)

if __name__ == '__main__':
    main()
        
        
    