from __future__ import print_function

import os,sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import subprocess as sp
from glob import glob
from dask import compute, delayed
from distributed import Client
from dask_jobqueue import SLURMCluster
import tarfile
from .db_utils import add_files_to_tarfile
import multiprocessing as mp

def run_cmd(cmd):
    p = sp.Popen(cmd, shell=True)
    p.communicate()
    ret = int(p.returncode)
    if ret != 0:
        return " ".join(cmd), ret
    return ret

def gen_params_from_folder(indir, outdir, mode='multiprocessing', 
                            overwrite=False, nopdb=False, mol2gen_app=None, 
                            multimol2=False, infer_atomtypes=False, queue='cpu'):
    pattern = os.path.join(indir, "*.mol2")
    mol2fns = sorted(glob(pattern))
    print("Number of mol2fns:", len(mol2fns))

    if mode == 'slurm':
        cluster = SLURMCluster(cores=1, processes=1, memory="2GB", 
                            queue=queue, job_name="gen_params_worker",
                            walltime="3:00:00", worker_extra_args=["--lifetime", "175m", "--lifetime-stagger", "4m"])

        n_workers = min( 300, int( len(mol2fns)*0.02 )+1 )
        if multimol2:
            n_workers = min( 300, int( len(mol2fns) ) )
        print(f"Using {n_workers} workers")
        cluster.adapt(minimum=0, maximum=n_workers, wait_count=400)
        client = Client(cluster)
        print(client.scheduler_info())
    if mol2gen_app is None:
        rosettahome = os.environ.get('ROSETTAHOME')
        if rosettahome is None:
            raise Exception("Error: neither of mol2gen_app or ROSETTAHOME is set.")
        print(f"ROSETTAHOME: {rosettahome}")
        mol2gen_app = os.path.join(rosettahome,
                                "source/scripts/python/public/generic_potential/mol2genparams.py")
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Made dir: {outdir}")

    joblist = []
    for i, mol2fn in enumerate(mol2fns):
        if (i+1)%10000 == 0:
            print(f"Adding {i+1}/{len(mol2fns)}")
        mol2name = os.path.basename(mol2fn).replace(".mol2","")
        paramsfn = os.path.join(outdir, mol2name+".params")
        if os.path.exists(paramsfn) and not overwrite:
            print("%s exists, continue.."%paramsfn)
            continue
        resname = "LG1" #this is 3 letter code, doesn't matter, typenm matters
        res_type_name = "automol2name"
        cmd = ["python3", "-W", "ignore", mol2gen_app,  "-s", mol2fn, "--outdir", outdir, 
                    "--resname", resname, "--typenm", res_type_name, "--rename_atoms"]
        if nopdb:
            cmd.append("--no_pdb")
        if multimol2:
            cmd.append("--multimol2")
        if infer_atomtypes:
            cmd.append("--infer_atomtypes")
        
        cmd = " ".join(cmd)
        #print(cmd)
        if mode == 'slurm':
            joblist.append( client.submit(run_cmd, cmd) )
        elif mode == 'local':
            joblist.append( delayed(run_cmd)(cmd) )
        elif mode == 'multiprocessing':
            joblist.append( cmd )
        elif mode == 'debug':
            print(cmd)
        else:
            print(cmd)
            retval = run_cmd(cmd)
            print(retval)

    if mode == 'slurm':
        print("Use dask SLURMCluster..")
        print("Total jobs: ", len(joblist))
        results = client.gather(joblist)
        print(results)
        client.close()
    elif mode == 'local':
        print("Use dask client local..")
        print("Total jobs: ", len(joblist))
        results = client.compute(joblist)
        print(results)
    elif mode == 'multiprocessing':
        ncpus = 24
        print(f"Use multiprocessing, ncpus {ncpus}..")
        print("Total jobs: ", len(joblist))
        with mp.Pool(ncpus) as pool:
            results = pool.map(run_cmd, joblist)
        print(results)

def gen_tarparams_from_list(paramsfns, outfn, overwrite=False):
    if not overwrite and os.path.exists(outfn):
        print(f"{outfn} exists, skip.")
        return

    n = add_files_to_tarfile(paramsfns, outfn, backup=False, overwrite=True)


def ligandlist_from_tarparamsfn(intarfn, outfn):
    ligandlines = []
    with tarfile.open(intarfn, 'r') as intar:
        raw_members = intar.getmembers()
        for member in raw_members:
            try:
                if not member.isfile():
                    continue
                if ".params" not in member.name:
                    continue
                if os.path.basename(member.name).startswith("._"):
                    continue
                f = intar.extractfile(member)
                lines = f.readlines()
                if len(lines) == 0:
                    print(f"Warning: {intarfn} {member.name} is empty, skip")
                    continue
                line = lines[0].strip()
                ligandID = line.decode().split()[1]
                ligandlines.append(ligandID+"\n")
            except Exception as e:
                print(f"Failed {intarfn} {member.name}")
                raise e

    with open(outfn, 'w') as outf:
        outf.writelines(ligandlines)
        print("Saved: %s"%outfn)


if __name__ == "__main__":
    pass
