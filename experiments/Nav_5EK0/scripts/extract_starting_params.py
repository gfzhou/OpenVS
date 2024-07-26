import os
import random
import numpy as np
import multiprocessing as mp
import time
from openvs.utils.utils import load_configfn
from openvs.utils.db_utils import extract_tarparams_to_folder, extract_tarparams_to_folder_wrapper, dataframe2dict
from distributed import Client
from dask_jobqueue import SLURMCluster
import pandas as pd

RANDOM_SEED = 1568

def load_full_database(index_dir):
    time0 = time.time()
    fulldb = None
    for l1 in "CDE":
        for l2 in "ABCDEFG":
            time1 = time.time()
            tid2l = f"{l1}{l2}"
            print(f"Loading {tid2l}")
            featherfn = os.path.join(index_dir, f"{tid2l}_zincids_paramsfn_index.feather")
            if not os.path.exists(featherfn):
                print(f"{featherfn} doesn't exists, skip")
                continue
            db_subset = pd.read_feather(featherfn)
            time2 = time.time()
            time_diff = time2-time1
            print(f"{tid2l} size is {len(db_subset)}, loading time is {time_diff} seconds")
            if fulldb is None:
                fulldb = db_subset
            else:
                fulldb = fulldb.append(db_subset, ignore_index=True)
    time2 = time.time()
    time_diff = time2-time0
    print(f"Loading full db took {time_diff} seconds")
    print(f"Size of full db is {len(fulldb)}")
    return fulldb
            
def sample_from_dataframe(df, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))

def random_select_cluster_center_params(index_dir, outdir, ntrain=500000, ntest=1000000, nval=1000000):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    fulldb = load_full_database(index_dir)

    ntotal = ntrain+ntest+nval
    if ntotal > len(fulldb):
        print(f"total sample size ({ntotal}) is larger than full database size ({len(fulldb)}).")
        raise
    sample_db_all = fulldb.sample(n=ntotal, replace=False, random_state=RANDOM_SEED, ignore_index=True)
    dbfn = os.path.join(outdir, "train_zincids_1.feather")
    sample_db = sample_db_all[:ntrain]
    sample_db.reset_index(inplace=True)
    sample_db.to_feather(dbfn)
    print(f"Saved: {dbfn}")

    dbfn = os.path.join(outdir, "test_zincids.feather")
    sample_db = sample_db_all[ntrain:ntrain+ntest]
    sample_db.reset_index(inplace=True)
    sample_db.to_feather(dbfn)
    print(f"Saved: {dbfn}")

    dbfn = os.path.join(outdir, "validation_zincids.feather")
    sample_db = sample_db_all[ntrain+ntest:ntotal]
    sample_db.reset_index(inplace=True)
    sample_db.to_feather(dbfn)
    print(f"Saved: {dbfn}")


def map_tarparamsfn_to_zincids( indexfn, tarparams_dir ):

    df = pd.read_feather(indexfn)
    paramsfn2zincids = dataframe2dict(df, "paramsfn", "zincid")
    tarparams2zincids = {}
    for paramsfn in paramsfn2zincids:
        fields = paramsfn.split("_") #clusters_CA_cluster_centroids_0_985.params
        tid2l = fields[1]
        tarparamsfname = "_".join( fields[:-1] ) + ".tar"
        tarparamsfn = os.path.join(tarparams_dir, tid2l, tarparamsfname)
        tarparams2zincids.setdefault(tarparamsfn, []).extend( paramsfn2zincids[paramsfn] )

    return tarparams2zincids

def extract_paramsfns_to_folder(indexfn:str, outdir:str, 
                                tarparams_dir:str, mode="multiprocess" ):
    n_proc=16
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if mode =="slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="3GB",
                queue='short', job_name="extract_worker", extra=["--no-nanny", "--no-bokeh"],
                walltime="1:00:00")
        cluster_obj.adapt(minimum=0, maximum=100, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
        print(client.scheduler_info())
    jobs_args = []
    failed_3d_zincids = []
    n_extracted = 0
    joblist = []
    tarparams2zincids = map_tarparamsfn_to_zincids(indexfn, tarparams_dir)
    for tarparamfn in tarparams2zincids:
        zincids = tarparams2zincids[tarparamfn]
        if os.path.exists(tarparamfn):
            inargs = (zincids, tarparamfn, outdir, "")
            if mode == "multiprocess":
                jobs_args.append( inargs ) 
            elif mode =="slurm":
                joblist.append( client.submit(extract_tarparams_to_folder_wrapper, inargs) )
            else:
                n_extracted += extract_tarparams_to_folder(zincids, tarparamfn, outdir, "")
        else:
            print(f"{len(zincids)} molecules don't have params files in database.")
            
    if mode=="multiprocess":
        print(f"Number jobs: {len(jobs_args)}")
        with mp.Pool(n_proc) as p:
            results = p.map(extract_tarparams_to_folder_wrapper, jobs_args)
        n_extracted = np.sum(results)
    elif mode=="slurm":
        print("Number jobs:", len(joblist))
        results = client.gather(joblist)
        n_extracted = np.sum(results)

    print(f"Done extracting 3d structures to {outdir}")
    print(f"Extracted {n_extracted} params files in database.")

def select_starting_dataset(configfn):
    config = load_configfn(configfn)
    outdir = os.path.join(config['project_tempdir'], "params" )
    index_dir = config['index_dir_params']
    ntrain = config['train_size']
    ntest = config['test_size']
    nval = config['val_size']
    random_select_cluster_center_params(index_dir, outdir, ntrain, ntest, nval)

def extract_train(configfn):
    config = load_configfn(configfn)
    tarparams_dir = config['params_basedir']
    time1 = time.time()
    train_indexfn = os.path.join(config['project_tempdir'], "params", "train_zincids_1.feather")
    outdir = os.path.join(config['project_tempdir'], "params", "train1_params" )
    extract_paramsfns_to_folder(train_indexfn, outdir, tarparams_dir, mode="multiprocess")
    time2 = time.time()
    time_diff = time2-time1
    print(f"Extract train params took {time_diff} seconds.")

def extract_test(configfn):
    config = load_configfn(configfn)
    time1 = time.time()
    tarparams_dir = config['params_basedir']
    indexfn = os.path.join(config['project_tempdir'], "params", "test_zincids.feather")
    outdir = os.path.join(config['project_tempdir'], "params", "test_params")
    extract_paramsfns_to_folder(indexfn, outdir, tarparams_dir, mode="multiprocess")
    time2 = time.time()
    time_diff = time2-time1
    print(f"Extract test params took {time_diff} seconds.")

def extract_validation(configfn):
    config = load_configfn(configfn)
    time1 = time.time()
    tarparams_dir = config['params_basedir']
    indexfn = os.path.join(config['project_tempdir'], "params", "validation_zincids.feather")
    outdir = os.path.join(config['project_tempdir'], "params", "validation_params")
    extract_paramsfns_to_folder(indexfn, outdir, tarparams_dir, mode="multiprocess")
    time2 = time.time()
    time_diff = time2-time1
    print(f"Extract validation params took {time_diff} seconds.")


def main():
    configfn = "../config_clusterdb.json"
    select_starting_dataset(configfn)
    extract_train(configfn)
    extract_test(configfn)
    extract_validation(configfn)

if __name__ == '__main__':
    main()

