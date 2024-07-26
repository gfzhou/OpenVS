import os,sys
from glob import glob
import subprocess as sp
from distributed import Client
from dask_jobqueue import SLURMCluster
import multiprocessing as mp
from pathlib import Path

def run_cmd(cmd, wdir):
    os.makedirs(wdir, exist_ok=True)
    p = sp.Popen(cmd, shell=True, cwd=wdir)
    p.communicate()
    return p.returncode

def eval_pdb(inargs):
    pdbfn, paramsfn, outpath = inargs
    rosettahome = os.environ.get('ROSETTAHOME')
    if rosettahome is None:
        raise Exception("Error: env variable ROSETTAHOME is not set.")
    print(f"ROSETTAHOME: {rosettahome}")
    app = os.path.join(rosettahome,
                            "source/bin/rosetta_scripts.linuxgccrelease")
    cwd = os.getcwd()
    hbfn = os.path.join(cwd, "flags.hb")
    xmlfn = os.path.join(cwd, "eval.xml")
    wcst = "0.0"
    prefix = "eval."
    cmd = [
    app,
    "-extra_res_fa", paramsfn,
    "-s", pdbfn,
    "-constrain_relax_to_start_coords",
    f"@{hbfn}",
    "-gen_potential",
    "-overwrite",
    "-beta_cart",
    "-no_autogen_cart_improper",
    "-parser:protocol", xmlfn,
    f"-score:set_weights coordinate_constraint {wcst}",
    "-out:levels", "all:300",
    "-out:prefix", prefix,
    "-out:file:scorefile", f"{prefix}score.sc",
    "-missing_density_to_jump",
    "-ignore_unrecognized_res",
    ">&", "/dev/null"
    ]
    print(" ".join(cmd))
    
    return run_cmd(" ".join(cmd), outpath)

def eval_screening_pdbfns(inpath, outrootpath, mode='slurm'):
    
    pdbfns = []
    patt = os.path.join(inpath, "????", "????", "*.pdb")
    pdbfns += sorted( glob(patt) )
    print("Number of pdb files to evaluate", len(pdbfns))
    if mode =="slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="3GB",
                queue='cpu', job_name="eval_worker",
                walltime="36:00:00")
        cluster_obj.adapt(minimum=0, maximum=500, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
    
    paramspath = os.path.join(Path(__file__).parents[1], "params_new")
    joblist = []
    for pdbfn in pdbfns:
        recid = os.path.dirname(pdbfn).split("/")[-2]
        trgid = os.path.dirname(pdbfn).split("/")[-1]
        outpath = os.path.join(outrootpath, recid, trgid)
        pdbfname = os.path.basename(pdbfn).replace(".pdb", "")
        outpdbfn = os.path.join(outpath, f"eval.{pdbfname}_0001.pdb")
        if os.path.exists(outpdbfn):
            continue
        os.makedirs(outpath, exist_ok=True)
        paramsfn = os.path.join(paramspath, f"{trgid}_ligand.am1bcc.params")
        inargs = (pdbfn, paramsfn, outpath)
        if mode == "slurm":
            joblist.append(client.submit(eval_pdb, inargs))
        elif mode == 'mp':
            joblist.append(inargs)
        else:
            ret = eval_pdb(inargs)
            print(ret)
    
    if mode == "slurm":
        print("Number of slurm jobs:", len(joblist))
        client.gather(joblist)
    elif mode == 'mp':
        print("Number of mp jobs:", len(joblist))
        with mp.Pool(16) as pool:
            results = pool.map(eval_pdb, joblist)
        print(results)
        
def main():
    inpath = os.path.join(os.getcwd(), "decoys_screening_complex_pdbs")
    outpath = "eval_screening_cpp_relax_lig_simple_no_cst"
    mode = 'mp'
    eval_screening_pdbfns(inpath, outpath, mode=mode)

if __name__ == "__main__":
    main()
