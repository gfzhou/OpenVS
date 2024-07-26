import os
import sys
import time
import pandas as pd
import subprocess as sp
from glob import glob
from distributed import Client
from dask_jobqueue import SLURMCluster
from multiprocessing import Pool
from openvs.utils.utils import smiles_to_binary_fingerprints, load_configfn


def load_cluster_fp_db(fps_path):
    fulldb = None
    for l1 in "CDE":
        for l2 in "ABCDEFG":
            time1 = time.time()
            tid2l = f"{l1}{l2}"
            print(f"Loading {tid2l}")
            featherfn = os.path.join(fps_path, f"{tid2l}.fp.feather")
            if not os.path.exists(featherfn):
                print(f"{featherfn} doesn't exists, skip")
                continue
            db_subset = pd.read_feather(featherfn)
            time2 = time.time()
            time_diff = time2-time1
            print(
                f"{tid2l} size is {len(db_subset)}, loading time is {time_diff} seconds")
            if fulldb is None:
                fulldb = db_subset
            else:
                fulldb = fulldb.append(db_subset, ignore_index=True)
    return fulldb

def load_top_prediction_db(top_prediction_path):
    dbfns = [os.path.join(top_prediction_path, "all.top.feather"),
             os.path.join(top_prediction_path, "all.random.feather")]
    fulldb = []
    for dbfn in dbfns:
        if not os.path.exists(dbfn):
            print(f"{dbfn} not exist.")
            return None
        fulldb.append(pd.read_feather(dbfn))
    
    fulldb = pd.concat(fulldb, ignore_index=True)
    if 'fp_binary' not in fulldb.columns:
        print("No fp_binary column in top prediction files")
        return None
    return fulldb
    

def collect_info_block_helper(inargs):
    return collect_info_block(*inargs)


def collect_info_block(infn, dbfn, index_column="zincid", fps_column="fp_hexstring"):
    df = pd.read_feather(infn)
    if index_column in df.columns:
        pass
    elif index_column not in df.columns and 'ligandname' in df.columns:
        df[index_column] = df['ligandname'].map(lambda x: x.split('_')[0])
    elif index_column not in df.columns and 'ligandname' not in df.columns:
        raise Exception(f"Neither {index_column} nor ligandname exists.")
    keys = set(df[index_column])

    db = pd.read_feather(dbfn)
    db.set_index(index_column, inplace=True)
    keys_common = set(db.index) & keys
    if not keys_common:
        return None
    df = db.loc[keys_common][['smiles', fps_column]]
    df.reset_index(inplace=True)
    return df


def concat_dfs(dfs):
    return pd.concat(dfs, ignore_index=True)


def collect_info_helper(inpdfn, dbfns, index_column="molecule_id", fps_column="fp_hexstring", mode="slurm"):
    print(f"Total of {len(dbfns)} db files found")
    if mode == "slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="5GB",
                                   queue='cpu', job_name="collect",
                                   walltime="2:50:00")
        ncpus = min(300, len(dbfns))
        cluster_obj.adapt(minimum=0, maximum=ncpus, wait_count=400)
        client = Client(cluster_obj)
        print(f"Using slurm clusters, {ncpus} cores:")
        print(client.scheduler_info())

    joblist = []
    dfs = []
    for i, dbfn in enumerate(dbfns):
        if not os.path.exists(dbfn):
            continue

        inargs = (inpdfn, dbfn, index_column, fps_column)
        if mode == "slurm":
            joblist.append(client.submit(collect_info_block_helper, inargs))
        elif mode == "multiprocessing":
            joblist.append(inargs)
        else:
            print(f"Working on {i+1}/{len(dbfns)}")
            ret_df = collect_info_block_helper(inargs)
            if ret_df is None:
                continue
            dfs.append(ret_df)

    if mode == "slurm":
        print("Number slurm jobs:", len(joblist))
        dfs_raw = client.gather(joblist)
        dfs = []
        for df in dfs_raw:
            if df is None:
                continue
            dfs.append(df)
    elif mode == "multiprocessing":
        print("Number mutiprocessing jobs:", len(joblist))
        with Pool(processes=20) as pool:
            dfs_raw = pool.map(collect_info_block_helper, joblist)
        dfs = []
        for df in dfs_raw:
            if df is None:
                continue
            dfs.append(df)
        print("Done with multiprocessing jobs.")

    df_info_all = concat_dfs(dfs)
    if 'zinc_id' in df_info_all.columns:
        df_info_all.rename(columns={'zinc_id': index_column}, inplace=True)
    print(df_info_all.head())

    indf = pd.read_feather(inpdfn)
    if index_column in indf.columns:
        pass
    elif index_column not in indf.columns and 'ligandname' in indf.columns:
        indf[index_column] = indf['ligandname'].map(lambda x: x.split('_')[0])
    elif index_column not in indf.columns and 'ligandname' not in indf.columns:
        raise Exception(f"Neither {index_column} nor ligandname exists.")
    zincids1 = set(indf[index_column])
    zincids2 = set(df_info_all[index_column])
    diff = zincids1-zincids2
    print(f"Number of different zincids: {len(diff)}")
    if zincids2 != zincids1:
        print(
            f"Warning: Not all entries are found in database. found {len(zincids2&zincids1)}/{len(zincids1)}")
    return df_info_all

def collect_info_fulldb(inpdfn, fps_path, index_column="zincid", mode="slurm"):
    indf = pd.read_feather(inpdfn)
    if index_column in indf.columns:
        pass
    elif index_column not in indf.columns and 'ligandname' in indf.columns:
        pass
    elif index_column not in indf.columns and 'ligandname' not in indf.columns:
        raise Exception(f"Neither {index_column} nor ligandname exists.")

    pattern = os.path.join(fps_path, "??", "*.feather")
    dbfns = sorted(glob(pattern))
    fps_column="fp_hexstring"
    return collect_info_helper(inpdfn, dbfns, index_column, fps_column, mode)

def collect_info_realdb(inpdfn, fps_path, index_column="molecule_id", mode="slurm"):
    indf = pd.read_feather(inpdfn)
    assert index_column in indf.columns or 'ligandname' in indf.columns, f"Cannot find column {index_column} nor ligandname in {inpdfn}"

    pattern = os.path.join(fps_path, "Enamine_REAL_*", "*.feather")
    dbfns = sorted(glob(pattern))
    fps_column="fp_binary"
    return collect_info_helper(inpdfn, dbfns, index_column, fps_column, mode)


def collect_info_zinc22db(inpdfn, fps_path, index_column="molecule_id", mode="slurm"):
    indf = pd.read_feather(inpdfn)
    assert index_column in indf.columns or 'ligandname' in indf.columns, f"Cannot find column {index_column} nor ligandname in {inpdfn}"

    pattern = os.path.join(fps_path, "zinc-22*", "*.feather")
    dbfns = sorted(glob(pattern))
    fps_column="fp_binary"
    return collect_info_helper(inpdfn, dbfns, index_column, fps_column, mode)


def add_fps_column(inpdfn, outpdfn, infulldb, molid_header="zincid", recompute_fps=False, backup_path=None):
    if backup_path is not None:
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        if os.path.exists(outpdfn):
            cmd = f"cp {outpdfn} {backup_path}"
            p = sp.popen(cmd, shell=True)
            p.communicate()
            if int(p.returncode) != 0:
                raise Exception(f"{cmd} failed")

    df = pd.read_feather(inpdfn)
    if molid_header in df.columns:
        pass
    elif molid_header not in df.columns and 'ligandname' in df.columns:
        df[molid_header] = df['ligandname'].map(lambda x: x.split('_')[0])
    elif molid_header not in df.columns and 'ligandname' not in df.columns:
        raise Exception(f"Neither {molid_header} nor ligandname exists.")
    fulldb = infulldb.drop_duplicates(
        subset=molid_header, keep="first", ignore_index=True)
    fulldb.set_index(molid_header, inplace=True)

    df.set_index(molid_header, inplace=True)
    index_common = list(set(fulldb.index) & set(df.index))
    if len(index_common) < len(set(df.index)):
        print(
            f"Warning: only found {len(index_common)}/{len(set(df.index))} entries!")
    df_new = df.loc[index_common]
    df_new['smiles'] = fulldb.loc[index_common]['smiles']
    if recompute_fps:
        df_new['fp_binary'] = smiles_to_binary_fingerprints(
            df_new['smiles'], radius=2, nBits=1024, useFeature=True, useChirality=True)
    else:
        df_new['fp_binary'] = fulldb.loc[index_common]['fp_binary']
    df_new.reset_index(inplace=True, drop=False)
    df_new.to_feather(outpdfn)
    print(f"Number of molecules: {len(df_new)}")
    print(f"Saved: {outpdfn}")


def add_fps_column_from_input(inpdfn, outpdfn, fp_column="fp_binary", backup_path=None):
    df = pd.read_feather(inpdfn)
    assert 'smiles' in df.columns, "Cannot find smiles column in {inpdfn}"
    if backup_path is not None:
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
        if os.path.exists(outpdfn):
            cmd = f"cp {outpdfn} {backup_path}"
            p = sp.popen(cmd, shell=True)
            p.communicate()
            if int(p.returncode) != 0:
                raise Exception(f"{cmd} failed")
    df_new = df.copy()
    df_new['fp_binary'] = smiles_to_binary_fingerprints(
        df['smiles'], radius=2, nBits=1024, useFeature=True, useChirality=True)
    df_new.to_feather(outpdfn)
    print(f"Number of molecules: {len(df_new)}")
    print(f"Saved: {outpdfn}")


def add_fps_column_from_input_warpper(inargs):
    add_fps_column_from_input(*inargs)


def augment_results_fns_from_input(infns, mode="slurm", backup_path=None, overwrite=False):
    if len(infns) == 0:
        return
    if len(infns) == 1:
        infn = infns[0]
        outfn = infn.replace(".feather", "")+".aug.feather"
        if os.path.exists(outfn) and not overwrite:
            return
        add_fps_column_from_input(infn, outfn, fp_column="fp_binary", backup_path=backup_path)
    else:
        if mode == "slurm":
            cluster_obj = SLURMCluster(cores=1, processes=1, memory="10GB",
                                       queue='cpu', job_name="add_fps",
                                       walltime="12:00:00")
            ncpus = min(10, len(infns))
            cluster_obj.adapt(minimum=0, maximum=ncpus, wait_count=400)
            client = Client(cluster_obj)
            print(f"Using slurm clusters, {ncpus} cores:")
            print(client.scheduler_info())
        joblist = []
        for infn in infns:
            outfn = infn.replace(".feather", "")+".aug.feather"
            if os.path.exists(outfn) and not overwrite:
                continue
            if mode == "slurm":
                inargs = (infn, outfn, "fp_binary", backup_path)
                joblist.append(client.submit(
                    add_fps_column_from_input_warpper, inargs))
            else:
                add_fps_column_from_input_warpper(inargs)

        if mode == "slurm":
            print("Number slurm jobs:", len(joblist))
            print(client.gather(joblist))
            
def augment_fn(infn, configfn, dbtype="cluster", mode="slurm", overwrite=False):
    config = load_configfn(configfn)
    resultspath = config['result_path']
    fps_path = config['fps_path']
    backup_path = os.path.join(resultspath, "backups")
    fulldb = None
    molid_header = "molecule_id"
    recompute_fps = False
    if dbtype == "cluster":
        fulldb = load_cluster_fp_db(fps_path)
        molid_header = "zincid"
        recompute_fps = True
    elif dbtype == "full":
        molid_header = "zincid"
        recompute_fps = True
        pass
    elif dbtype == "real" or dbtype=="zinc22":
        molid_header = "molecule_id"
    elif dbtype == "None":
        infns = []
        infns.append(infn)
        augment_results_fns_from_input(
            infns, mode="slurm", backup_path=None, overwrite=False)
        return
    else:
        raise Exception("FPs database not loaded.")

    outfn = infn.replace(".feather", "")+".aug.feather"
    if not overwrite and os.path.exists(outfn):
        print(f"{outfn} exists, skip.")
        return
    if not os.path.exists(infn):
        print(f"Warning: {infn} is not found, skip!")
        return
    print(f"Augmenting {infn}")
    if dbtype == "full":
        fulldb = collect_info_fulldb(infn, fps_path, "zincid", mode)
    elif dbtype == "real":
        fulldb = collect_info_realdb(infn, fps_path, "molecule_id", mode)
    elif dbtype == "zinc22":
        fulldb = collect_info_zinc22db(infn, fps_path, "molecule_id", mode)
    add_fps_column(infn, outfn, fulldb, molid_header, recompute_fps, backup_path)

def augment_result_fns(i_iter, configfn, dbtype="cluster", mode="slurm", overwrite=False):
    config = load_configfn(configfn)
    resultspath = config['result_path']
    proj_name = config['project_name']
    filenames = [f'{proj_name}_train{i_iter}_vs_results.feather']
    fps_path = config['fps_path']
    
    if i_iter == 1:
        filenames += [f"{proj_name}_test_vs_results.feather",
                      f"{proj_name}_validation_vs_results.feather"]
    backup_path = os.path.join(resultspath, "backups")
    fulldb = None
    molid_header = "molecule_id"
    recompute_fps = False
    if dbtype == "cluster":
        fulldb = load_cluster_fp_db(fps_path)
        molid_header = "zincid"
        recompute_fps = True
    elif dbtype == "full":
        molid_header = "zincid"
        recompute_fps = True
        pass
    elif dbtype == "real" or dbtype == "zinc22":
        molid_header = "molecule_id"
    elif dbtype == "None":
        infns = []
        for fname in filenames:
            infns.append(os.path.join(resultspath, fname))
        augment_results_fns_from_input(
            infns, mode="slurm", backup_path=None, overwrite=False)
        return
    else:
        raise Exception("FPs database not loaded.")
    for fname in filenames:
        infn = os.path.join(resultspath, fname)
        outfn = infn.replace(".feather", "")+".aug.feather"
        if not overwrite and os.path.exists(outfn):
            print(f"{outfn} exists, skip.")
            continue
        if not os.path.exists(infn):
            print(f"Warning: {infn} is not found, skip!")
            continue
        print(f"Augmenting {infn}")
        if dbtype == "full":
            fulldb = collect_info_fulldb(infn, fps_path, "zincid", mode)
        elif dbtype == "real":
            fulldb = collect_info_realdb(infn, fps_path, "molecule_id", mode)
        elif dbtype == "zinc22":
            prediction_path = os.path.join(config['prediction_path'], f"model_{i_iter-1}_prediction")
            top_prediction_path = os.path.join(prediction_path, "top_predictions")
            print(f"Try to load top prediction file from last iter {i_iter-1}")
            fulldb = load_top_prediction_db(top_prediction_path)
            if fulldb is None:
                print("Failed loading top prediction files")
                print("Try to load the whole zinc22 db")
                fulldb = collect_info_zinc22db(infn, fps_path, "molecule_id", mode)
        add_fps_column(infn, outfn, fulldb, molid_header, recompute_fps, backup_path)

def main():
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    mode = "slurm"
    if i_iter == 1:
        configfn = "../config_clusterdb.json"
        dbtype="cluster"
    else:
        configfn = "../config_real_db.json"
        dbtype="real"

    augment_result_fns(i_iter, configfn, dbtype=dbtype,
                       mode=mode, overwrite=False)

if __name__ == "__main__":
    main()
