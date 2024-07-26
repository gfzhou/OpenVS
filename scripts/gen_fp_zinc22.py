import os,sys
from glob import glob
import tarfile

# import scientific py
import numpy as np
import pandas as pd
# rdkit stuff
from rdkit import Chem
from rdkit.Chem import AllChem

from distributed import Client, as_completed, wait
from dask_jobqueue import SLURMCluster
import multiprocessing as mp


def valid_molid(molid):
    if molid.startswith('ZINC') or molid.startswith('P') or molid.startswith('Z'):
        return True
    return False

def convert_zinc22_tgz_to_fpfn_wrapper(inargs):
    return convert_zinc22_tgz_to_fpfn(*inargs)

def convert_zinc22_tgz_to_fpfn(infn, fpfn, relpath="", nBits=1024, radius=2, useFeature=True, useChirality=True, overwrite=True):
    if os.path.exists(fpfn) and not overwrite:
        print(f"{fpfn} exists, skip")
        return 0

    if infn.endswith('gz'):
        rmode = "r:gz"
    elif infn.endswith('tar'):
        rmode = "r"
    else:
        raise Exception(f"{infn} format not supported")

    data = []
    n_failed=0
    with tarfile.open(infn, rmode) as intar:
        members = intar.getmembers()
        for member in members:
            try:
                if not member.isfile():
                    continue
                if not member.name.endswith("mol2"):
                    continue
                if os.path.basename(member.name).startswith("._"):
                    continue
                f = intar.extractfile(member)
                mol2block = f.read()
                if len(mol2block) == 0:
                    print(f"Warning: {infn} {member.name} is empty, skip")
                    continue
                m = Chem.MolFromMol2Block(mol2block, removeHs=True)
                if m is None:
                    print(f"{infn}, {member.name} is failed in RDKit")
                    n_failed+=1
                    continue
                    #raise Exception(f"{infn}, {member.name} is failed in RDKit")
                molid = m.GetProp('_Name')
                if not valid_molid(molid):
                    raise Exception(f"Cannot find valid molecule id, current molecule id {molid}")
                smi = Chem.MolToSmiles(m)
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, 
                            nBits=nBits, useFeatures=useFeature, useChirality=useChirality)

                data.append([molid, smi, fp.ToBinary()])
            except Exception as e:
                print(f"Failed {infn} {member.name}")
                raise e
    if len(data) == 0:
        print(f"All {n_failed} molecules failed in {infn}, don't save the feather file.")
        return n_failed
    if relpath != "":
        df = pd.DataFrame(data, columns=['molecule_id', 'smiles', 'fp_binary']) 
        df["relpath"] = [relpath]*len(df)
    else:
        df = pd.DataFrame(data, columns=['molecule_id', 'smiles', 'fp_binary'])

    df.to_feather(fpfn)
    print(f"Saved: {fpfn}")
    return n_failed

def convert_zinc22_tgzfns(inbasedir, outbasedir, mode="slurm"):
    if mode=="slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="10GB",
                queue='cpu', job_name="gen_fp",
                walltime="12:00:00")
        cluster_obj.adapt(minimum=0, maximum=400, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
        print(client.scheduler_info())
    elif mode == "mp":
        print("Using multiprocessing.")
    elif mode == "debug":
        pass
    else:
        raise Exception("Invalid running mode.")

    patt = os.path.join(inbasedir, "**/*.tgz")
    allfns = glob(patt, recursive=True)
    print("Total number of tgz files:", len(allfns))
    
    argslist = []
    results = []
    oformat = "feather"
    delimiter = " "
    nBits = 1024
    radius = 2
    useFeature=True
    useChirality=True
    overwrite = False
    for infn in allfns:
        relpath = os.path.relpath(infn, inbasedir)
        infname=os.path.basename(infn)
        outdir = os.path.join(outbasedir, os.path.dirname(relpath))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        infname = infname.replace(".tgz", "")
        outfname = f"{infname}.fp.{oformat}"
        outfn = os.path.join(outdir, outfname)
        if os.path.exists(outfn) and not overwrite: continue
        in_args = (infn, outfn, relpath, nBits, radius, useFeature, useChirality, overwrite)
        if mode == "slurm":
            results.append( client.submit(convert_zinc22_tgz_to_fpfn_wrapper, in_args) )
            argslist.append(in_args)
        elif mode == "mp":
            argslist.append(in_args)
        elif mode == "debug":
            print(in_args)
            convert_zinc22_tgz_to_fpfn_wrapper(in_args)
            sys.exit()

    print("Number of jobs:",len(argslist))
            
    if mode=="slurm":
        retvals = client.gather(results)
        retvals = np.array(retvals)
        I = np.where(retvals>0)[0]
        content = ['gen_fp_zinc22\n']
        for i in I:
            content.append(f"{argslist[i][0]} {argslist[i][1]}\n")
        failfn = 'gen_fp_failed.txt'
        with open(failfn, 'a') as outf:
            outf.writelines(content)
        print(f"updated {failfn}")
        
    elif mode == "mp":
        ncpus=20
        with mp.Pool(ncpus) as pool:
            retvals = pool.map(convert_zinc22_tgz_to_fpfn_wrapper, argslist)


if __name__ == "__main__":
    mode = "mp"
    inbasedir = "../databases/zinc/zinc22/zinc22"
    outbasedir = "../databases/zinc/zinc22/fingerprints/"
    convert_zinc22_tgzfns(inbasedir, outbasedir, mode)
