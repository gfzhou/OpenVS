import os,sys
import orjson
from glob import glob

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def write_sbatchfn(sbatchfn, submitfn, joblistfn, n_jobs, cwd="./", job_name="arrayjobs", queue='cpu'):

    if queue == 'cpu':   
        sbatch_content = """#!/bin/bash
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=500m
#SBATCH --time 6:30:00
#SBATCH --job-name={jobname}
#SBATCH -o output.{jobname}.log

source ~/.bashrc
conda activate openvs
CMD=$(head -$SLURM_ARRAY_TASK_ID {joblistfn} | tail -1)
exec ${{CMD}}
""".format(jobname=job_name, joblistfn=joblistfn, cwd=cwd)
 
    with open(sbatchfn, 'w') as outf:
        outf.write(sbatch_content)
    print("Wrote: %s"%sbatchfn)

    submit_content = """#!/bin/bash
sbatch -a 1-{} {}
""".format( n_jobs,os.path.basename(sbatchfn) )
    with open(submitfn, 'w') as outf:
        outf.write(submit_content)
    print("Wrote: %s"%submitfn)

def gen_joblist(smipath, outdir):
    joblist = []
    obabel_app = os.path.join(os.getcwd(), "run_obabel.sh")
    pattern = os.path.join(smipath, "prot", "*.smi")
    smifns = sorted(glob(pattern))
    for smifn in smifns:
        prefix = os.path.basename(smifn).replace(".smi", "")
        tempprefix = os.path.join(outdir, prefix+".temp")
        outmol2fn = os.path.join(outdir, f"{prefix}.mol2")
        cmd = f"{obabel_app} {smifn} {tempprefix} {outmol2fn} 0\n"
        joblist.append(cmd)
    pattern = os.path.join(smipath, "obabel", "*.smi")
    smifns = sorted(glob(pattern))
    for smifn in smifns:
        prefix = os.path.basename(smifn).replace(".smi", "")
        tempprefix = os.path.join(outdir, prefix)
        outmol2fn = os.path.join(outdir, f"{prefix}.mol2")
        cmd = f"{obabel_app} {smifn} {tempprefix} {outmol2fn} 1\n"
        joblist.append(cmd)

    return joblist

def gen3d_joblist_from_smipath(smipath, mol2outdir, jobname="gen3d", overwrite=False):
    queue="cpu"
    joblist_lines = gen_joblist(smipath, mol2outdir)
    
    joblist_fn = os.path.join("./", f"{jobname}.joblist" )

    with open(joblist_fn, 'w') as outf:
        outf.writelines(joblist_lines)
    print("Wrote: %s"%joblist_fn)
    n_jobs = len(joblist_lines)

    sbatchfn = os.path.join( "./", f"sbatch_arrayjobs.{jobname}.sh" )
    submitfn = os.path.join( "./", f"submit_arrayjobs.{jobname}.sh" )
    joblistfn = joblist_fn
    
    write_sbatchfn(sbatchfn, submitfn, joblistfn, n_jobs, mol2outdir, jobname, queue)


def gen3d_real_db(i_iter, configfn, debug=False):
    config = load_configfn(configfn)
    if debug:
        outdir = "debug_gen3d"
        os.makedirs(outdir, exist_ok=True)
    if not debug:
        outdir = os.path.join(config['mol2_path'], f"mmff94_mol2s_iter{i_iter+1}")
        os.makedirs(outdir, exist_ok=True)
    smipath = os.path.join(os.getcwd(), f"smiles_iter{i_iter+1}")
    jobname = "gen3d"
    
    gen3d_joblist_from_smipath(smipath, outdir, jobname)


if __name__ == "__main__":
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    configfn = os.path.join("../", "config_real_db.json" )
    debug=False

    gen3d_real_db(i_iter, configfn, debug)
