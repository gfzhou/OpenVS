import os,sys
import subprocess as sp
import multiprocessing as mp
import tarfile
import pandas as pd
import orjson

def run_cmd(cmd, wdir, logfn):
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    with open(logfn, 'w') as logfile:
        p = sp.Popen(cmd, cwd=wdir, stdout=logfile)
        p.communicate()
    return p.returncode

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def extract_params_from_tarfn(tarfn, ligandnames, outdir, overwrite=True):
    ligandnames = set(ligandnames)
    extracted = set([])
    with tarfile.open(tarfn, "r") as tarfh:
        for member in tarfh.getmembers():
            if len(extracted) == len(ligandnames):
                break
            if not member.isfile():
                continue
            if ".params" not in member.name:
                continue
            if os.path.basename(member.name).startswith("._"):
                continue
            f = tarfh.extractfile(member)
            lines = f.readlines()
            ligandname = lines[0].decode().strip().split()[1]
            if ligandname in ligandnames:
                extracted.add(ligandname)
                outfn = os.path.join(outdir, f"{ligandname}.params")
                if os.path.exists(outfn) and not overwrite:
                    continue
                with open(outfn, 'w') as outfh:
                    lines = [l.decode() for l in lines]
                    outfh.writelines(lines)
    print(f"Extracted {len(extracted)}/{len(ligandnames)}")
    return extracted

def save_top_N_params_from_df(df, config, ntop, outdir, overwrite=True):

    df.sort_values(by='dG', inplace=True, ignore_index=True)
    print(df.head())
    iter2ligandnames = {}
    iter2descriptions = {}
    
    for irow in range(ntop):
        iter2ligandnames.setdefault( df.iloc[irow]["i_iter"], [] ).append(df.iloc[irow]['ligandname'])
        iter2descriptions.setdefault( df.iloc[irow]["i_iter"], [] ).append(df.iloc[irow]['description'])

    ligand_list=[]
    params_path = os.path.join(config['project_tempdir'], 'params')
    inargs = []
    for i_iter in iter2ligandnames:
        print(i_iter)
        if os.path.exists(os.path.join(params_path, f"train{i_iter}_params_0.tar")):
            params_fn = os.path.join(params_path, f"train{i_iter}_params_0.tar")
            prefix = f"train{i_iter}"
        elif os.path.exists(os.path.join(params_path, f"{i_iter}_params_0.tar")):
            params_fn = os.path.join(params_path, f"{i_iter}_params_0.tar")
            prefix = i_iter
        print("params_fn: ",params_fn )
    
        ligandnames = iter2ligandnames[i_iter]
        outpath = os.path.join(outdir, f"params_top{ntop}", f"{prefix}")
        os.makedirs(outpath, exist_ok=True)
        inargs.append((params_fn, ligandnames, outpath, overwrite))
    
    ncpus= 12
    with mp.Pool(ncpus) as pool:
        extracted_lignames = pool.starmap(extract_params_from_tarfn , inargs)
    
    for lignames in extracted_lignames:
        ligand_list.extend(list(lignames))

    liglistfn = os.path.join(outdir, f"ligandlist_top{ntop}.txt")
    with open(liglistfn, 'w') as outfh:
        outfh.write('\n'.join(ligand_list))
    print(f"Saved: {liglistfn}")

def save_top_N_params(prefixes, configfn, ntop, outdir, overwrite=True):
    config = load_configfn(configfn)
    summary_path = config['result_path']

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    df_all = None
    for prefix in prefixes:
        summary_fn = os.path.join(summary_path, f"{config['project_name']}_{prefix}_vs_results.aug.feather")
        if not os.path.exists(summary_fn):
            print(f"Cannot find {summary_fn}")
            continue
        df = pd.read_feather(summary_fn)
        df['i_iter'] = [prefix]*len(df)
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)
        
    save_top_N_params_from_df(df_all, config, ntop, outdir, overwrite)

def run_save_top_N_params():
    configfn = "../config_zinc22_db.json"
    config = load_configfn(configfn)
    ntop = 1000
    prefix="substructure"
    outdir = os.path.join(config['project_tempdir'], 'top_params', f'{prefix}')
    overwrite = True
    prefixes = [prefix]
    save_top_N_params(prefixes, configfn, ntop=ntop, outdir=outdir, overwrite=overwrite)
    
def run_save_top_cluster_params():
    configfn = "../config_zinc22_db.json"
    config = load_configfn(configfn)
    ntop = 1000
    prefix = "substructure"
    outdir = os.path.join(config['project_tempdir'], 'top_params_cluster', f'{prefix}')
    
    clusterfn = f"cluster.{prefix}.300000.feather"
    df = pd.read_feather(clusterfn)
    df.sort_values(by='dG', inplace=True, ignore_index=True)
    df.drop_duplicates(subset=['clusterid'], keep='first',inplace=True, ignore_index=True)
    overwrite = True
    
    save_top_N_params_from_df(df, config, ntop, outdir, overwrite)
    
if __name__ == "__main__":
    #run_save_top_cluster_params()
    run_save_top_N_params()
