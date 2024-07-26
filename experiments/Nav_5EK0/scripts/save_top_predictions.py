import os,sys
import pandas as pd
from glob import glob
import orjson
import numpy as np
from distributed import Client, as_completed, wait
from dask_jobqueue import SLURMCluster
from tqdm import tqdm
import multiprocessing  as mp
from time import time


def ceildiv(a,b):
    return -(a//-b)

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def save_top_tranche_wrapper(inargs):
    tid2l, database_dir, prediction_dir, outfn, ntops, fileformat=inargs
    return save_top_tranche(tid2l, database_dir, prediction_dir, outfn, ntops, fileformat)

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
            
    return dbfns

def read_all_dbfns(infns, mode='mp'):
    # a list of feather files
    ncpus=4
    if mode == 'mp':
        with mp.Pool(ncpus) as pool:
            df = pool.map(pd.read_feather, infns)
    else:
        df = [pd.read_feather(infn) for infn in infns]
    df = pd.concat(df, ignore_index=True)
    return df

def save_top_tranche(tid2l, database_dir, prediction_dir, outfn, ntops=10000, fileformat="feather"):
    if os.path.exists(outfn):
        return
    pattern = os.path.join(database_dir, tid2l, f"*.{fileformat}")
    dbfns = glob(pattern)
    top_df = None
    for dbfn in dbfns:
        dbfname = os.path.basename(dbfn)
        tid6l = dbfname.split(".")[0]
        dbfidx = dbfname.split(".")[1]
        pred_fname = dbfname.rstrip(f".{fileformat}")+f".predict.{fileformat}"
        pred_fn = os.path.join(prediction_dir, tid2l, pred_fname)
        if not os.path.exists(pred_fn):
            raise Exception(f"{pred_fn} doesn't exist!")
        if fileformat == "feather":
            df_db = pd.read_feather(dbfn)
            df_pred = pd.read_feather(pred_fn)
        elif fileformat == "csv":
            df_db = pd.read_csv(dbfn)
            df_pred = pd.read_csv(pred_fn)
        else:
            raise Exception(f"Format {fileformat} is wrong.")
        df_db["tid6l"] = [tid6l]*len(df_db)

        merged_df = pd.merge(left=df_db, right=df_pred, left_on="zinc_id", right_on="ZINCID")
        merged_df.drop(["zinc_id"], inplace=True, axis=1)

        if top_df is None:
            top_df = merged_df
        else:
            top_df = top_df.append(merged_df, ignore_index=True)
            top_df = top_df.nlargest(ntops, 'p_hits', keep='all')
          
    top_df.reset_index(drop=True, inplace=True)

    if fileformat == "feather":
        top_df.to_feather(outfn)
    elif fileformat == "csv":
        top_df.to_csv(outfn)
    print(f"Saved: {outfn}")

def save_top_files_helper(inargs):
    return save_top_files(*inargs)

def save_top_files(dbfns, outfn, ntops=10000, fileformat="feather", molid_header="molecule_id", excl_dbfns=None, cutoff=None, cutoff_ratio=None):
    if os.path.exists(outfn):
        print(f"{outfn} exists.")
        return len(pd.read_feather(outfn))

    excl_zincids = None
    if excl_dbfns is not None:
        for dbfn in excl_dbfns:
            df = pd.read_feather(dbfn)
            if molid_header not in df:
                print(f"molecule id column {molid_header} is not in {dbfn}, skip")
                continue
            if excl_zincids is None:
                excl_zincids = set(df[molid_header])
            else:
                excl_zincids.update(set(df[molid_header]))

    top_df = None
    for dbfn in dbfns:
        if not os.path.exists(dbfn):
            raise Exception(f"{dbfn} doesn't exist!")
        if fileformat == "feather":
            df = pd.read_feather(dbfn)
        elif fileformat == "csv":
            df = pd.read_csv(dbfn)
        else:
            raise Exception(f"Format {fileformat} is wrong.")

        if top_df is None:
            top_df = df
        else:
            top_df = pd.concat([top_df, df], ignore_index=True )

        top_df.drop_duplicates(subset=molid_header, keep='first', inplace=True, ignore_index=True)
        if excl_zincids is not None:
            top_df.reset_index(drop=True, inplace=True)
            top_df.set_index(molid_header, inplace=True)
            kept_index = set(top_df.index) - excl_zincids
            top_df = top_df.loc[kept_index]
            top_df.reset_index(inplace=True)
        if cutoff is None:
            top_df = top_df.nlargest(ntops, 'p_hits', keep='all')
        else:
            top_df = top_df.loc[top_df['p_hits']>=cutoff]
            top_df = top_df.nlargest(ntops, 'p_hits', keep='all')

    top_df.reset_index(drop=True, inplace=True)
    
    p_hits_min = np.min(top_df['p_hits'])
    if p_hits_min<0.5:
        print(f"Warning: smallest p_hits is {p_hits_min} in top {ntops} hits.")
    if fileformat == "feather":
        top_df.to_feather(outfn)
    elif fileformat == "csv":
        top_df.to_csv(outfn)
    print(f"Saved: {outfn}, n_entry: {len(top_df)}")
    return len(top_df)

def save_random_files_helper(inargs):
    return save_random_files(*inargs)

def save_random_files(dbfns, outfn, nkeep=10000, fileformat="feather", molid_header="molecule_id", excl_dbfns=None, cutoff=None):
    if os.path.exists(outfn):
        print(f"{outfn} exists.")
        return

    excl_zincids = None
    if excl_dbfns is not None:
        for dbfn in excl_dbfns:
            df = pd.read_feather(dbfn)
            if molid_header not in df:
                print(f"molecule id column {molid_header} is not in {dbfn}, skip")
                continue
            if excl_zincids is None:
                excl_zincids = set(df[molid_header])
            else:
                excl_zincids.update(set(df[molid_header]))

    df_keep = None
    nsample = ceildiv(nkeep, len(dbfns))
    if nsample < 1:
        raise Exception(f"nsample per file is less than 1!")
    for dbfn in dbfns:
        if not os.path.exists(dbfn):
            raise Exception(f"{dbfn} doesn't exist!")
        if fileformat == "feather":
            df = pd.read_feather(dbfn)
        elif fileformat == "csv":
            df = pd.read_csv(dbfn)
        else:
            raise Exception(f"Format {fileformat} is wrong.")

        if excl_zincids is not None:
            df.reset_index(drop=True, inplace=True)
            df.set_index(molid_header, inplace=True)
            kept_index = list( set(df.index) - excl_zincids )
            df = df.loc[kept_index]
            df.reset_index(inplace=True)

        if df_keep is None:
            if cutoff is not None:
                df = df.loc[df['p_hits']>=cutoff]
            df_keep = df.sample( min(nsample, len(df)), replace=False, ignore_index=True )
        else:
            if cutoff is not None:
                df = df.loc[df['p_hits']>=cutoff]
            df_sample = df.sample( min(nsample, len(df)), replace=False, ignore_index=True )
            df_keep = pd.concat([df_keep, df_sample], ignore_index=True)

    # just to make sure there is no duplicates
    df_keep.drop_duplicates(subset=molid_header, keep='first', inplace=True, ignore_index=True)
    if fileformat == "feather":
        df_keep.to_feather(outfn)
    elif fileformat == "csv":
        df_keep.to_csv(outfn)
    print(f"Saved: {outfn}, n_entry: {len(df_keep)}")
    return len(df_keep)


def save_top_all_zinc22db(i_iter, configfn, mode, dockflex=False, batchsize=None, cutoff=None, cutoff_ratio=None):
    config = load_configfn(configfn)
    ntotal = config['train_size']
    if dockflex:
        ntotal=ntotal//2+int(0.02*ntotal)
    else:
        ntotal = int(ntotal*1.02)
        
    ntop_ratio = 0.5
    ntops=int(ntotal*ntop_ratio)
    nrandom=int(ntotal*(1-ntop_ratio))
    print(f"Save top ranked:{ntops}, save random: {nrandom}, total {ntotal}.")
    if cutoff is not None:
        print(f"Using p_hits cutoff {cutoff:.2f}.")

    fileformat="feather"
    molid_header="molecule_id"
    prediction_path = config['prediction_path']
    prediction_path = os.path.join(prediction_path, f"model_{i_iter}_prediction")
    print(f"Prediction path: {prediction_path}")
    if mode =="slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="10GB",
                queue='cpu', job_name="save_top_worker",
                walltime="12:00:00")
        cluster_obj.adapt(minimum=0, maximum=300, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
        print(client.scheduler_info())

    pattern = os.path.join(prediction_path, "**", "*.feather")
    allfns = glob(pattern, recursive=True)
    subdirs = set([os.path.basename(os.path.dirname(fn)) for fn in allfns])
    print("subdir names:", subdirs)
    outdir = os.path.join(prediction_path, "top_predictions")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    joblist = []
    topfns = []
    randomfns = []
    excl_dbfns = None
    # save top ranked molecules
    for subdir in subdirs:
        pattern = os.path.join(prediction_path, subdir, "**", "*.feather")
        dbfns = sorted(glob(pattern, recursive=True))
        if len(dbfns) == 0:
            print(f"No prediction found for {subdir}")
            continue
        if batchsize is None: 
            outfn = os.path.join(outdir, f"{subdir}.top.all.feather")
            topfns.append(outfn)
            if os.path.exists(outfn):
                print(f"{outfn} exists, skip.")
                continue
            inargs = (dbfns, outfn, ntops, fileformat, molid_header, excl_dbfns, cutoff, cutoff_ratio)
            if mode =="slurm":
                joblist.append( client.submit(save_top_files_helper, inargs) )
            else:
                save_top_files_helper(inargs)

        else:
            indices = np.arange(0, len(dbfns), batchsize)
            for i, idx in enumerate(indices):
                dbfns_batch = dbfns[idx:idx+batchsize]
                outfn = os.path.join(outdir, f"{subdir}.top.{i}.feather")
                topfns.append(outfn)
                if os.path.exists(outfn):
                    print(f"{outfn} exists, skip.")
                    continue
                inargs = (dbfns_batch, outfn, ntops, fileformat, molid_header, excl_dbfns, cutoff, cutoff_ratio)
                if mode =="slurm":
                    joblist.append( client.submit(save_top_files_helper, inargs) )
                else:
                    save_top_files_helper(inargs)

    # random sample molecules
    for subdir in subdirs:
        pattern = os.path.join(prediction_path, subdir, "**", "*.feather")
        dbfns = sorted(glob(pattern, recursive=True))
        if len(dbfns) == 0:
            print(f"No prediction found for {subdir}")
            continue
        if batchsize is None and nrandom>0: 
            outfn = os.path.join(outdir, f"{subdir}.random.all.feather")
            randomfns.append(outfn)
            if os.path.exists(outfn):
                print(f"{outfn} exists, skip.")
                continue
            inargs = (dbfns, outfn, nrandom, fileformat, molid_header, excl_dbfns, cutoff)
            if mode =="slurm":
                joblist.append( client.submit(save_random_files_helper, inargs) )
            else:
                save_random_files_helper(inargs)
        if batchsize is not None and nrandom>0:
            indices = np.arange(0, len(dbfns), batchsize)
            for i, idx in enumerate(indices):
                dbfns_batch = dbfns[idx:idx+batchsize]
                outfn = os.path.join(outdir, f"{subdir}.random.{i}.feather")
                randomfns.append(outfn)
                if os.path.exists(outfn):
                    print(f"{outfn} exists, skip.")
                    continue
                inargs = (dbfns_batch, outfn, nrandom, fileformat, molid_header, excl_dbfns, cutoff)
                if mode =="slurm":
                    joblist.append( client.submit(save_random_files_helper, inargs) )
                else:
                    save_random_files_helper(inargs)
        
    if mode == 'slurm':
        print("Number jobs:", len(joblist))
        print(client.gather(joblist))

    test_fn = config["test_datafn"]
    val_fn = config["val_datafn"]
    excl_dbfns = [test_fn, val_fn]
    train_fn_path = config["train_data_path"]
    for i in range(1, i_iter+1):
        train_fn = os.path.join(train_fn_path, f"{config['prefix']}_train{i}_vs_results.aug.feather")
        if not os.path.exists(train_fn):
            print(f"Cannot find {train_fn}")
            train_fn = os.path.join(train_fn_path, f"{config['prefix']}_train{i}_vs_results.feather")
            if not os.path.exists(train_fn):
                print(f"Cannot find {train_fn}")
                continue
        excl_dbfns.append(train_fn)
    print(f"Exclude molecules from: {excl_dbfns}")
    outfn_all = os.path.join(outdir, "all.top.feather")
    n_save_top = save_top_files(topfns, outfn_all, ntops, fileformat, molid_header, excl_dbfns, cutoff) 
    if n_save_top < ntops:
        print(f"Warning: {outfn_all} size is {n_save_top}, which is smaller than specified size: {ntops}")
    if nrandom>0:
        excl_dbfns.append(outfn_all)
        outfn_all_random = os.path.join(outdir, "all.random.feather")
        n_save_random = save_random_files(randomfns, outfn_all_random, nrandom, fileformat, molid_header, excl_dbfns, cutoff)
        if n_save_random < nrandom:
            print(f"Warning: {outfn_all_random} size is {n_save_random}, which is smaller than specified size: {nrandom}")

def save_topN_from_dbfns(dbfns, outfn, molid_header, nkeep, batchsize, disable_progress_bar=True):
    indices = np.arange(0, len(dbfns), batchsize)
    df_top = None
    for idx in tqdm(indices, disable=disable_progress_bar):
        t0 = time()
        df_batch = read_all_dbfns(dbfns[idx:idx+batchsize], mode='mp')
        t1 = time()
        print(f"Reading took {t1-t0}s")
        if df_top is None:
            df_top = df_batch
            continue
        df_top = pd.concat([df_top, df_batch], ignore_index=True)
        t2 = time()
        print(f"concat took {t2-t1}s")
        df_top = df_top.sort_values(by=["p_hits"], ascending=False).head(nkeep)
        t3 = time()
        print(f"sort_values took {t3-t2}s")
    df_top.drop_duplicates(subset=molid_header, keep='first', inplace=True, ignore_index=True)
    t4 = time()
    print(f"drop_duplicates took {t4-t3}s")
    df_top.reset_index(drop=True, inplace=True)
    df_top.to_feather(outfn)
    print(f"Saved {outfn}")


if __name__ == "__main__":
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    configfn = os.path.join("../", "config_zinc22_db.json" )
    mode='mp'
    cutoff = 1E-6
    cutoff_ratio=0.05
    dockflex = False
    save_top_all_zinc22db(i_iter, configfn, mode, dockflex, batchsize=50, cutoff=cutoff, cutoff_ratio=cutoff_ratio)
    
