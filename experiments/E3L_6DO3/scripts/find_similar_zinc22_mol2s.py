import os
import orjson
import pandas as pd
from glob import glob

from rdkit import Chem

from distributed import Client
from dask_jobqueue import SLURMCluster

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def fetch_dbfns(dbtype, db_path):
    if dbtype=="cluster":
        dbfn_pattern = os.path.join(db_path, "*.feather")
        dbfns = sorted( glob(dbfn_pattern) )
    elif dbtype == "full":
        tids = []
        dbfns = []
        for l1 in "BCDEFGHIJ":
           for l2 in "ABCDEFGHIJ":
                tids.append(f"{l1}{l2}")
        for tid in tids:
            dbfn_pattern = os.path.join(db_path, tid, "*.feather")
            dbfns.extend( sorted( glob(dbfn_pattern) ) )
    elif dbtype == "real":
        dbfn_pattern = os.path.join(db_path, "Enamine_REAL*", "*.feather")
        dbfns = sorted(glob(dbfn_pattern))
    elif dbtype == "zinc22":
        dbfn_pattern = os.path.join(db_path, "zinc-22*", "*.feather")
        dbfns = sorted(glob(dbfn_pattern))
            
    return dbfns

def save_substructure_match_wrapper(inargs):
    return save_substructure_match(*inargs)

def save_substructure_match(smarts_incl, smarts_excl, dbfn, outfn):
    patts_incl = [Chem.MolFromSmarts(smarts) for smarts in smarts_incl]
    patts_excl = [Chem.MolFromSmarts(smarts) for smarts in smarts_excl]
    
    df = pd.read_feather(dbfn)
    smiles = list( df['smiles'] )
    sel_ndx = []
    for i, smi in enumerate(smiles):
        skip = False
        m = Chem.MolFromSmiles(smi)
        for patt_excl in patts_excl:
            if m.HasSubstructMatch(patt_excl):
                skip=True
                break
        if skip: continue
        
        keep = False
        for patt_incl in patts_incl:
            if m.HasSubstructMatch(patt_incl):
                keep = True
                break
        if not keep: continue
        sel_ndx.append(i)
    
    if len(sel_ndx) > 0:
        df_similar = df.iloc[sel_ndx]
        df_similar.reset_index(drop=True, inplace=True)
        df_similar.to_feather(outfn)
        print(f"Saved: {outfn}")
        return len(sel_ndx)
    
    return 0

def find_substructure_in_db(molid, smarts_incl, smarts_excl, configfn, outdirname="substructure_mols", use_slurm=True):
    config = load_configfn(configfn)
    database_type = config["database_type"]
    database_path = config["fps_path"]
    dbfns = fetch_dbfns(database_type, database_path)
    
    outdir = os.path.join(config["project_tempdir"], outdirname, molid)
    os.makedirs(outdir, exist_ok=True)
    if use_slurm:
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="5GB",
                    queue='cpu', job_name="find_similar",
                    walltime="12:00:00")
        cluster_obj.adapt(minimum=0, maximum=300, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
        print(client.scheduler_info())
    
    print("Predicting the database...")
    joblist = []
    print(f"Number of fingerprints files: {len(dbfns)}")
    for dbfn in dbfns:
        if not os.path.exists(dbfn):
            continue
        dbfname = os.path.basename(dbfn)
        outfname = ".".join(dbfname.split(".")[:-1] + ["substructure", "feather"])
        outfn = os.path.join(outdir, outfname)
        if os.path.exists(outfn):
            print(f"{outfn} exists.")
            continue
        inargs = (smarts_incl, smarts_excl, dbfn, outfn)
        if use_slurm:
            joblist.append( client.submit(save_substructure_match_wrapper, inargs) )
        else:
            ret = save_substructure_match_wrapper(inargs)
            print(ret)
    
    if use_slurm:
        print("Number jobs:", len(joblist))
        print(client.gather(joblist))


def find_substructure_match():
    configfn = "../config_zinc22_db.json"
    molid = "Z3009405982"
    use_slurm = True
    outdirname = "substructure_mols"
    smarts_incl = ["O=CNCc1cn(CC(=O)O)nn1"]
    smarts_excl = ["COC(=O)Cn1ccnn1"]
    find_substructure_in_db(molid, smarts_incl, smarts_excl, configfn, outdirname, use_slurm)
    
    
def main():
    find_substructure_match()

if __name__ == "__main__":
    main()