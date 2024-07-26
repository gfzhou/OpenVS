import os,sys,io
import orjson
import pandas as pd
import tarfile
from glob import glob
import subprocess as sp
import multiprocessing as mp
import numpy as np

from dask import compute, delayed
import dask.multiprocessing
from distributed import Client, as_completed, wait
from dask_jobqueue import SLURMCluster

def dataframe2dict(df, key_column, value_column):
    retdict={}
    if key_column not in df.columns:
        raise Exception(f"{key_column} is a wrong column name.")
    if value_column not in df.columns:
        raise Exception(f"{value_column} is a wrong column name.")
    for (index, data) in df.iterrows():
        retdict.setdefault(data[key_column], []).append(data[value_column])
    
    return retdict

def map_raw3dfn_to_zincids_wrapper(inargs):
    tid4l, dbdir, indexdir, zincids = inargs
    return map_raw3dfn_to_zincids(tid4l, dbdir, indexdir, zincids)

def map_raw3dfn_to_zincids_tid2l_wrapper(inargs):
    tid2l, dbdir, indexdir, zincids = inargs
    return map_raw3dfn_to_zincids_tid2l(tid2l, dbdir, indexdir, zincids)


def map_raw3dfn_to_zincids(tid4l:str, dbdir:str, indexdir:str, zincids:set ):
    pattern  = os.path.join(indexdir, tid4l[:2], f"{tid4l}*.json")
    db_indexfns = glob(pattern)
    zincids_to_3dfns = {}
    for db_indexfn in db_indexfns:
        with open(db_indexfn, 'rb') as infh:
            dbindex = orjson.loads(infh.read())
            zincids_found = zincids & dbindex.keys()
            for zincid in zincids_found:
                zincids_to_3dfns[zincid] = dbindex[zincid]
        if len(zincids_to_3dfns) == len(zincids):
            break
    
    mol2fn_to_zincids ={}
    for zincid in zincids_to_3dfns:
        mol2fn = zincids_to_3dfns[zincid]
        if mol2fn != "" and os.path.exists(mol2fn):
            mol2fn_to_zincids.setdefault(mol2fn, set([])).update([zincid])
        elif os.path.exists( os.path.join(dbdir, tid4l[:2], mol2fn) ):
            mol2fn_to_zincids.setdefault(os.path.join(dbdir, tid4l[:2], mol2fn), set([])).update([zincid])
        else:
            mol2fn_to_zincids.setdefault(f"NA_{tid4l}", set([])).update([zincid])
    
    return mol2fn_to_zincids

def map_raw3dfn_to_zincids_tid2l(tid2l:str, dbdir:str, indexdir:str, zincids:set ):
    pattern  = os.path.join(indexdir, tid2l, f"{tid2l}*.json")
    db_indexfns = glob(pattern)
    zincids_to_3dfns = {}
    for db_indexfn in db_indexfns:
        with open(db_indexfn, 'rb') as infh:
            dbindex = orjson.loads(infh.read())
            zincids_found = zincids & dbindex.keys()
            for zincid in zincids_found:
                zincids_to_3dfns[zincid] = dbindex[zincid]
        if len(zincids_to_3dfns) == len(zincids):
            break
    
    mol2fn_to_zincids ={}
    for zincid in zincids_to_3dfns:
        mol2fn = zincids_to_3dfns[zincid]
        if mol2fn != "" and os.path.exists(mol2fn):
            mol2fn_to_zincids.setdefault(mol2fn, set([])).update([zincid])
        elif os.path.exists( os.path.join(dbdir, tid2l, mol2fn) ):
            mol2fn_to_zincids.setdefault(os.path.join(dbdir, tid2l, mol2fn), set([])).update([zincid])
        else:
            mol2fn_to_zincids.setdefault(f"NA_{tid2l}", set([])).update([zincid])
    
    return mol2fn_to_zincids

def extract_tarmember_to_folder_wrapper(inargs):
    zincids, intarfn, outdir, extra, logpath = inargs
    return extract_tarmember_to_folder(zincids, intarfn, outdir, extra, logpath)

def extract_tarmember_to_folder( zincids, intarfn, outdir, extra="", logpath=""):
    n = 0
    extracted=set([])
    with tarfile.open(intarfn, 'r') as intarfh:
        for member in intarfh.getmembers():
            zincid = os.path.basename(member.name).split(".")[0]
            if zincid not in zincids:
                continue
            f = intarfh.extractfile(member)
            outfn = os.path.join(outdir, os.path.basename(member.name))
            if extra != "":
                outfn = outfn.replace(".mol2", "")+ f".{extra}.mol2"
            if os.path.exists(outfn):
                n += 1
                extracted.add(zincid)
                continue
            with open(outfn, 'w') as outfh:
                outfh.write(f.read().decode())
            n += 1
            if n == len(zincids):
                break

    if n != len(zincids):
        print(f"Warning: extract only {n}/{len(zincids)} members to {outdir}")
        if logpath != "":
            if not os.path.exists(logpath):
                os.makedirs(logpath)
            outfname = os.path.basename(intarfn)
            outfname = outfname.replace(".tar", "")
            outfname = outfname.replace(".tgz", "")
            outfname = outfname + ".failed.txt"
            
            outfn = os.path.join(logpath, outfname)
            content = []
            for zincid in set(zincids) - extracted:
                content.append(f"{zincid}\n")
            with open(outfn, 'w') as outfh:
                outfh.writelines(content)
            print(f"Saved: {outfn}")
    else:
        print(f"Successfully extracted {n} members to {outdir}")
    return n


def extract_tarparams_to_folder_wrapper(inargs):
    zincids, intarfn, outdir, extra = inargs
    return extract_tarparams_to_folder(zincids, intarfn, outdir, extra)

def extract_tarparams_to_folder( zincids, intarfn, outdir, extra=""):
    n = 0
    with tarfile.open(intarfn, 'r') as intarfh:
        for member in intarfh.getmembers():
            if not member.name.endswith(".params"):
                continue
            f = intarfh.extractfile(member)
            zincid = f.readlines()[0].decode().strip().split()[1].split("_")[0]
            if zincid not in zincids:
                continue
            f = intarfh.extractfile(member)
            outfn = os.path.join(outdir, os.path.basename(member.name))
            if extra != "":
                outfn = outfn.replace(".params", "")+ f".{extra}.params"
            if os.path.exists(outfn):
                n += 1
                continue
            with open(outfn, 'w') as outfh:
                outfh.write(f.read().decode())
            n += 1
            if n == len(zincids):
                break

    if n != len(zincids):
        print(f"Warning: extract only {n}/{len(zincids)} members to {outdir}")
    else:
        print(f"Successfully extracted {n} members to {outdir}")
    return n

# add tar members to another tar file
def add_tarmember_to_tarfile(zincids, intarfn, outtarfn, backup=True):
    if os.path.exists(outtarfn):
        tar_mode = "a"
        if backup:
            tarfn_dir = os.path.dirname(outtarfn)
            backup_dir = os.path.join(tarfn_dir, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            cmd = ["cp", outtarfn, backup_dir]
            p = sp.Popen(cmd)
            p.communicate()
    else:
        tar_mode = "w"
    n = 0
    with tarfile.open(outtarfn, tar_mode) as outtarfh:
        with tarfile.open(intarfn, 'r') as intarfh:
            for member in intarfh.getmembers():
                zincid = os.path.basename(member.name).split(".")[0]
                if zincid not in zincids:
                    continue
                f = intarfh.extractfile(member)
                outtarfh.addfile(member, f)
                n += 1
                if n == len(zincids):
                    break
    if n != len(zincids):
        print(f"Warning: added only {n}/{len(zincids)} members to {outtarfn}")
    else:
        print(f"Successfully added {n} members to {outtarfn}")
    return n

# add regular files to tar file
def add_files_to_tarfile(regularfns, outtarfn, backup=True, overwrite=False):
    if os.path.exists(outtarfn):
        tar_mode = "a"
        if backup:
            tarfn_dir = os.path.dirname(outtarfn)
            backup_dir = os.path.join(tarfn_dir, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            cmd = f"cp {outtarfn} {backup_dir}"
            p = sp.Popen(cmd, shell=True)
            p.communicate()
    else:
        tar_mode = "w"
    if overwrite:
        tar_mode = "w"

    n = 0
    with tarfile.open(outtarfn, tar_mode) as tarfh:
        for regularfn in regularfns:
            if not os.path.exists(regularfn):
                continue
            with open(regularfn, "rb") as infh:
                fname = os.path.basename(regularfn)
                info = tarfile.TarInfo(name=fname)
                data = infh.read()
                if len(data) == 0:
                    print(f"Warning: {regularfn} is empty, skip.")
                    continue
                info.size=len(data)
                tarfh.addfile(info, io.BytesIO(data))
                n += 1
    print(f"Added {n}/{len(regularfns)} files to {outtarfn}")
    return n


def extract_raw3dfns_for_dbfn(dbfn:str, outdir:str, 
                                raw3d_db_dir:str,
                                raw3d_dbindex_dir:str, 
                                topN=None,
                                logpath = "",
                                mode="multiprocess" ):
    n_proc=16
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if dbfn.endswith(".csv"):
        df = pd.read_csv(dbfn)
    elif dbfn.endswith(".feather"):
        df = pd.read_feather(dbfn)

    if topN is not None:
        df = df[:topN]
    print(f"Preparing {len(df)} 3d mols for {dbfn}")
    print(df.head())
    tid2zincid = {}
    zincid2smiles = {}
    zincid2tid = {}
    zincid_label =""
    if "ZINCID" in df.columns:
        zincid_label = "ZINCID"
    elif "zincid" in df.columns:
        zincid_label = "zincid"
    elif "zinc_id" in df.columns:
        zincid_label = "zinc_id"
    if zincid_label == "":
        raise Exception("Cannot find zincid column.")

    if mode =="slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="3GB",
                queue='cpu', job_name="extract_worker", extra=["--no-nanny", "--no-bokeh"],
                walltime="3:00:00")
        cluster_obj.adapt(minimum=0, maximum=100, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters:")
        print(client.scheduler_info())

    for i in range(len(df)):
        tid4l = df.iloc[i].tid6l[:4]
        tid2zincid.setdefault(tid4l, set([])).update( [ df.iloc[i][zincid_label]] )
        zincid2tid[ df.iloc[i][zincid_label] ] = tid4l

    jobs_args = []
    for tid4l in tid2zincid:
        zincids = tid2zincid[tid4l]
        jobs_args.append( (tid4l, raw3d_db_dir, raw3d_dbindex_dir, zincids) )
    with mp.Pool(n_proc) as p:
        results = p.map(map_raw3dfn_to_zincids_wrapper, jobs_args)

    mol2_to_zincids = {}
    zincids_found = set([])
    for result in results:
        for mol2fn in result:
            mol2_to_zincids.setdefault( mol2fn, set([]) ).update(result[mol2fn])
            zincids_found.update(result[mol2fn])
            
    print(f"Found {len(zincids_found)}/{len(df)} molecules")
    if len(df) > len(zincids_found) and logpath:
        df.set_index(zincid_label, inplace=True)
        diff_ids = set(df.index) - zincids_found
        df_diff = df.loc[diff_ids]
        df_diff.reset_index(inplace=True)
        outfn = os.path.join(logpath, "failed.feather")
        df_diff.to_feather(outfn)
        print(f"Saved: {outfn}")

    jobs_args = []
    failed_3d_zincids = []
    n_extracted = 0
    joblist = []
    for mol2fn in mol2_to_zincids:
        if os.path.exists(mol2fn):
            zincids = mol2_to_zincids[mol2fn]
            tid4l = os.path.basename(mol2fn).split(".")[0]
            if mode == "multiprocess":
                inargs = (zincids, mol2fn, outdir, tid4l, logpath)
                jobs_args.append( inargs ) 
            elif mode =="slurm":
                inargs = (zincids, mol2fn, outdir, tid4l, logpath)
                joblist.append( client.submit(extract_tarmember_to_folder_wrapper, inargs) )
            else:
                n_extracted += extract_tarmember_to_folder(zincids, mol2fn, outdir, tid4l, logpath)
        elif mol2fn.startswith("NA"):
            print(f"{len(zincids)} molecules don't have 3d structures in database.")
        elif not os.path.exists(mol2fn):
            print(f"{mol2fn} doesn't exist.")
            
    if mode=="multiprocess":
        print(f"Number jobs: {len(jobs_args)}")
        with mp.Pool(n_proc) as p:
            results = p.map(extract_tarmember_to_folder_wrapper, jobs_args)
        n_extracted = np.sum(results)
    elif mode=="slurm":
        print("Number jobs:", len(joblist))
        results = client.gather(joblist)
        n_extracted = np.sum(results)

    print(f"Done extracting 3d structures to {outdir}")
    print(f"Extracted {n_extracted} of {len(zincids_found)} structures found in database.")


if __name__ == "__main__":
    pass
