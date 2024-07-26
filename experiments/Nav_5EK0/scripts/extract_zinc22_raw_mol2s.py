import os,sys
import orjson
import pandas as pd
import multiprocessing as mp
import numpy as np

from distributed import Client
from dask_jobqueue import SLURMCluster
from openvs.utils.db_utils import extract_tarmember_to_folder, extract_tarmember_to_folder_wrapper

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def extract_mol2s_from_zinc22(infofns,
                            outdir:str, 
                            raw3d_db_dir:str, 
                            logpath = "",
                            mode="mp" ):
    
    os.makedirs(outdir, exist_ok=True)
    df = []
    for infofn in infofns:
        if not os.path.exists(infofn):
            print(f"{infofn} not exist, skip.")
        df.append(pd.read_feather(infofn))
    if len(df) == 0:
        print("No info file was found. Stop.")
        return
    
    df = pd.concat(df, ignore_index=True)
    n_proc = 24
    zincid_label =""
    if "molecule_id" in df.columns:
        zincid_label = "molecule_id"
    if "ZINCID" in df.columns:
        zincid_label = "ZINCID"
    elif "zincid" in df.columns:
        zincid_label = "zincid"
    elif "zinc_id" in df.columns:
        zincid_label = "zinc_id"
    if zincid_label == "":
        raise Exception("Cannot find zincid column.")

    df.drop_duplicates(subset=zincid_label, keep='first', inplace=True, ignore_index=True)
    print(f"Extracting {len(df)} 3d mols...")
    
    if mode =="slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="1GB",
                queue='cpu', job_name="extract_worker", 
                walltime="3:00:00")
        cluster_obj.adapt(minimum=0, maximum=100, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
        print(client.scheduler_info())

    jobs_args = []
    tgzfn_to_molids = {}
    for i in range(len(df)):
        print(f"{raw3d_db_dir}")
        print(f"{df.iloc[i]['relpath']}")
        
        tgzfn = os.path.join(raw3d_db_dir, df.iloc[i]['relpath'])
        if not os.path.exists(tgzfn):
            raise Exception(f"{tgzfn} not exist.")
        tgzfn_to_molids.setdefault( tgzfn, set([]) ).update([df.iloc[i][zincid_label]])
    print(f"Extracting from {len(tgzfn_to_molids)} files")
    if logpath:
        os.makedirs(logpath, exist_ok=True)

    jobs_args = []
    n_extracted = 0
    joblist = []
    extra = ""
    for mol2fn in tgzfn_to_molids:
        zincids = tgzfn_to_molids[mol2fn]
        if mode == "mp":
            inargs = (zincids, mol2fn, outdir, extra, logpath)
            jobs_args.append( inargs ) 
        elif mode =="slurm":
            inargs = (zincids, mol2fn, outdir, extra, logpath)
            joblist.append( client.submit(extract_tarmember_to_folder_wrapper, inargs) )
        else:
            n_extracted += extract_tarmember_to_folder(zincids, mol2fn, outdir, extra, logpath)

    if mode=="mp":
        print(f"Number jobs: {len(jobs_args)}")
        with mp.Pool(n_proc) as p:
            results = p.map(extract_tarmember_to_folder_wrapper, jobs_args)
        n_extracted = np.sum(results)
    elif mode=="slurm":
        print("Number jobs:", len(joblist))
        results = client.gather(joblist)
        n_extracted = np.sum(results)

    print(f"Done extracting 3d structures to {outdir}")
    print(f"Extracted {n_extracted} of {len(df)} structures in database.")

def main():
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    configfn = "../config_zinc22_db.json"
    config = load_configfn(configfn)
    zinc22_mol2path = config["zinc22_mol2_path"]
    outpath = os.path.join(config["mol2_path"], f"zinc22_mol2s_iter{i_iter+1}")
    logpath = "./logs"
    mode="mp"
    
    top_prediction_path = os.path.join(config['prediction_path'], f"model_{i_iter}_prediction", "top_predictions")
    topfn = os.path.join(top_prediction_path, "all.top.feather")
    randomfn = os.path.join(top_prediction_path, "all.random.feather")
    # make sure topfn before randomfn in the list
    infofns = [topfn, randomfn]
    extract_mol2s_from_zinc22(infofns, outpath, zinc22_mol2path, logpath, mode)

if __name__ == "__main__":
    main()
