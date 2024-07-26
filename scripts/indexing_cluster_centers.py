# This is a one-time-use scripts to index the cluster center structure am1bcc mol2s
import os,sys
import tarfile
import orjson
from glob import glob

from dask import compute, delayed
from distributed import Client, as_completed, wait
from dask_jobqueue import SLURMCluster
import pandas as pd
import multiprocessing as mp

def params_tar_to_dict(tarfn):
    db_dict = {}
    with tarfile.open(tarfn, 'r') as intarfh:
        for member in intarfh.getmembers():
            if not member.name.endswith(".params"):
                continue
            paramsfname = os.path.basename(member.name)
            f = intarfh.extractfile(member)
            zincid = f.readlines()[0].decode().strip().split()[1].split("_")[0]
            if not zincid.startswith("ZINC"):
                print(f"{zincid} is not a valid zincid in {tarfn}, {member.name}")
                raise
            db_dict[zincid] = paramsfname

    return db_dict
        

def index_tranche_am1bcc_mol2s(inargs):
    tid2l, outdir = inargs
    datadir = "/home/gzhou/virtual_screening/ZINC/make_on_demand_lead_like_smiles_Feb_27_2019_centroids_ECFP_am1bcc_mol2s"
    pattern = os.path.join(datadir, tid2l, "*.tar.gz")
    tarfns = sorted( glob(pattern) )
    raw_mol2_dbidx_dir = os.path.join("/home/gzhou/virtual_screening/dataset/ZINC/3D_druglike_tarmol2/index",
                            tid2l)
    pattern =  os.path.join(raw_mol2_dbidx_dir, "*.json")
    jsonfns = sorted(glob(pattern))
    zincids_to_rawmolfn = {}
    for jsonfn in jsonfns:
        with open(jsonfn, 'rb') as infh:
            index_db = orjson.loads(infh.read())
        zincids_to_rawmolfn.update(index_db)
    outdir = os.path.join(outdir, tid2l)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Made {outdir}")

    no_raw3dfn_list = []
    for tarfn in tarfns:
        cid1 = int(tarfn.split(".")[0].split("_")[-1])
        out_jsonfn = os.path.join(outdir, f"{tid2l}_{cid1}_zincid2raw3dfn.json")
        out_jsonfn2 = os.path.join(outdir, f"{tid2l}_{cid1}_clusterid2zincid.json")
        db_dict = {}
        db_dict2 = {}
        if os.path.exists(out_jsonfn):
            print(f"Warning: {out_jsonfn} exists, skip.")
            continue
        with tarfile.open(tarfn, 'r:gz') as intarfh:
            for member in intarfh.getmembers():
                if not member.name.endswith(".mol2"):
                    continue
                fields = os.path.basename(member.name).split(".")[0].split("_")
                cid1 = int(fields[-2])
                cid2 = int(fields[-1])
                clusterid = f"{tid2l}_{cid1}_{cid2}"

                f = intarfh.extractfile(member)
                zincid = f.readlines()[1].decode().strip()
                if not zincid.startswith("ZINC"):
                    print(f"{zincid} is not a valid zincid in {tarfn}, {member.name}")
                    raise
                if zincid in zincids_to_rawmolfn:
                    raw3dfn = zincids_to_rawmolfn[zincid]
                    tid4l = raw3dfn.split(".")[0]
                    chunkid = raw3dfn.split(".")[1]
                else:
                    print(f"Warning: {tarfn} {member.name} {zincid} doesn't have raw mol2 file!")
                    raw3dfn = ""
                    tid4l = ""
                    chunkid = ""
                    no_raw3dfn_list.append(f"{zincid}, {clusterid}\n")
                db_dict[zincid] = {"tid4l":tid4l,
                                    "raw3dfn": raw3dfn
                                    }
                db_dict2[clusterid] = zincid
        if len(db_dict) > 0:
            with open(out_jsonfn, 'wb') as outfh:
                outfh.write(orjson.dumps(db_dict, option=orjson.OPT_INDENT_2))
        print(f"Wrote: {out_jsonfn}")

        if len(db_dict2) > 0:
            with open(out_jsonfn2, 'wb') as outfh:
                outfh.write(orjson.dumps(db_dict2, option=orjson.OPT_INDENT_2))
        print(f"Wrote: {out_jsonfn2}")
    
    fail_fn = os.path.join(outdir, "fail_noraw3dfn.txt")
    with open(fail_fn, "w") as outfh:
        outfh.writelines(no_raw3dfn_list)
    print(f"Saved: {fail_fn}")

def index_tranche_params(inargs):
    tid2l, outdir = inargs
    datadir = "/home/gzhou/git/my_repos/OpenVS/databases/centroids"
    pattern = os.path.join(datadir, tid2l, "*.tar")
    tarfns = sorted( glob(pattern) )
    zincids_to_paramfn = {}

    os.makedirs(outdir, exist_ok=True)
    print(f"Made {outdir}")
    outfn = os.path.join(outdir, f"{tid2l}_zincids_paramsfn_index.feather")
    if os.path.exists(outfn):
        print(f"{outfn} exists, skip.")
        return
    tranche_db = {}
    for tarfn in tarfns:
        #print(tarfn)
        db_subset = params_tar_to_dict( tarfn )
        tranche_db.update( db_subset )
    if len(tranche_db) == 0:
        return
    df = pd.DataFrame.from_dict(tranche_db, orient="index",columns=['paramsfn'])
    df.reset_index(inplace=True)
    df.rename({'index':'zincid'}, inplace=True, axis='columns')
    df.to_feather(outfn)
    print(f"Saved: {outfn}")

# takes a tar params file and index it
def indexing_tar_params(inargs):
    tarfn, tid2l, outdir = inargs
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Made {outdir}")
    cid1 = int(tarfn.split(".")[0].split("_")[-1])
    out_jsonfn = os.path.join(outdir, f"{tid2l}_{cid1}_zincid2cids.json")
    if os.path.exists(out_jsonfn):
        print(f"Warning: {out_jsonfn} exists, skip.")
        return 1
    db_dict = {}
    with tarfile.open(tarfn, 'r') as intarfh:
        for member in intarfh.getmembers():
            if not member.name.endswith(".params"):
                continue
            fields = os.path.basename(member.name).split(".")[0].split("_")
            cid1 = int(fields[-2])
            cid2 = int(fields[-1])
            clusterid = f"{tid2l}_{cid1}_{cid2}"

            f = intarfh.extractfile(member)
            zincid = f.readlines()[0].decode().strip().split()[1].split("_")[0]
            if not zincid.startswith("ZINC"):
                print(f"{zincid} is not a valid zincid in {tarfn}, {member.name}")
                raise
            db_dict[zincid] = clusterid
    if len(db_dict) > 0:
        with open(out_jsonfn, 'wb') as outfh:
            outfh.write(orjson.dumps(db_dict, option=orjson.OPT_INDENT_2))
    print(f"Wrote: {out_jsonfn}")
    return 0

def indexing_all_am1bcc_mol2s(outdir):
    
    cluster_obj = SLURMCluster(cores=1, processes=1, memory="5GB",
                queue='dimaio', job_name="indexing", extra=["--no-nanny", "--no-bokeh"],
                walltime="10:00:00")
    cluster_obj.adapt(minimum=0, maximum=100, wait_count=400)
    client = Client(cluster_obj)
    print("Using slurm clusters:")
    print(client.scheduler_info())
    joblist = []
    for l1 in 'CDE':
        for l2 in 'ABCDEFG':
            tid2l = f"{l1}{l2}"
            inargs = (tid2l, outdir)
            joblist.append( client.submit(index_tranche_am1bcc_mol2s, inargs) )
    print("Number jobs:", len(joblist))
    results = client.gather(joblist)

def indexing_all_params(outrootdir):
    
    cluster_obj = SLURMCluster(cores=1, processes=1, memory="30GB",
                queue='dimaio', job_name="indexing", extra=["--no-nanny", "--no-bokeh"],
                walltime="10:00:00")
    cluster_obj.adapt(minimum=0, maximum=10, wait_count=400)
    client = Client(cluster_obj)
    print("Using slurm clusters:")
    print(client.scheduler_info())
    datadir = os.path.join( "/home/gzhou/virtual_screening/ZINC",
                "centroids_ECFP_params_Sep_2020" )
    joblist = []
    for l1 in 'CDE':
        for l2 in 'ABCDEFG':
            tid2l = f"{l1}{l2}"
            pattern = os.path.join(datadir, tid2l, "*.tar")
            tarfns = glob(pattern)
            outdir = os.path.join(outrootdir, tid2l)
            if os.path.exists(outdir):
                os.makedirs(outdir)
                print(f"Made {outdir}")
            for tarfn in tarfns:
                inargs = (tarfn, tid2l, outdir)
                joblist.append( client.submit(indexing_tar_params, inargs) )
    print("Number jobs:", len(joblist))
    results = client.gather(joblist)

def indexing_all_params_pandas(outdir, mode='mp'):
    
    if mode == 'slurm':
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="30GB",
                    queue='cpu', job_name="indexing", extra=["--no-nanny", "--no-bokeh"],
                    walltime="10:00:00")
        cluster_obj.adapt(minimum=0, maximum=20, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
        print(client.scheduler_info())
    joblist = []
    for l1 in 'CDE':
        for l2 in 'ABCDEFG':
            tid2l = f"{l1}{l2}"
            inargs = (tid2l, outdir)
            if mode == 'slurm':
                joblist.append( client.submit(index_tranche_params, inargs) )
            elif mode == 'mp':
                joblist.append(inargs)
    print("Number jobs:", len(joblist))
    if mode == 'slurm':
        results = client.gather(joblist)
    elif mode == 'mp':
        ncpus=24
        print(f"Using {ncpus} cores for multiprocessing..")
        with mp.Pool(ncpus) as pool:
            results = pool.map(index_tranche_params, joblist)
    
    print(results)

def debug(outdir):
    for l1 in 'C':
        for l2 in 'A':
            tid2l = f"{l1}{l2}"
            inargs = (tid2l, outdir)
            print(inargs)
            index_tranche_params(inargs)

if __name__ == '__main__':
    outdir = "/home/gzhou/git/my_repos/OpenVS/databases/centroids/index/"
    mode = 'mp'
    indexing_all_params_pandas(outdir, mode)
            

