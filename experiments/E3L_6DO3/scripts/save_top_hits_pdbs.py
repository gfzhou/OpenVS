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

def extract_silentfns_from_tarfn(tags, tarfn, tempdir):
    prefix2silentfns = {}
    tags_set = set(tags)
    with tarfile.open(tarfn, 'r') as tarfh:
        for member in tarfh.getmembers():
            if not member.name.endswith("out"):
                continue
            prefix = os.path.basename(member.name).replace(".out", "")
            if prefix not in tags_set:
                continue
            outsilentfn = os.path.join(tempdir, f"{prefix}.out")
            if os.path.exists(outsilentfn):
                print(f"Found: {outsilentfn}, skip")
                prefix2silentfns[prefix] = outsilentfn
                continue

            fh=tarfh.extractfile(member)
            content = fh.read()
            with open(outsilentfn, 'wb') as outfh:
                outfh.write(content)
            print(f"Wrote: {outsilentfn}")
            prefix2silentfns[prefix] = outsilentfn
            
            if len(prefix2silentfns) == len(tags):
                break

    return prefix2silentfns

def extract_pdbs_from_tar_resultfn(tar_result_fn, tarparamsfn, tempdir, outpdbpath, ligandnames, descriptions):
    if len(ligandnames) != len(descriptions):
        raise 
    lignames2descriptions = dict(zip(ligandnames, descriptions))
    temp_paramsdir = os.path.join(tempdir, "params")
    temp_silentdir = os.path.join(tempdir, "silentfns")

    if not os.path.exists(temp_paramsdir):
        os.makedirs(temp_paramsdir)
    if not os.path.exists(temp_silentdir):
        os.makedirs(temp_silentdir)

    #extract params files
    outtarparamsfn = os.path.join(temp_paramsdir, "ligand_params.tar")

    ligands_listfn = os.path.join(temp_paramsdir, "ligands_list.txt")
    logfn = os.path.join(temp_paramsdir, "extract.log")
    with open(ligands_listfn, 'w') as outfh:
        content = [ligandname+"\n" for ligandname in ligandnames]
        outfh.writelines(content)
        print("Saved: %s"%ligands_listfn)
    params_kept = set([])
    outtarfh = tarfile.open(outtarparamsfn, 'w')
    with tarfile.open(tarparamsfn, 'r') as tarfh:
        n = 0
        for member in tarfh.getmembers():
            if ".params" not in member.name:
                continue
            if "._" in member.name:
                continue
            fh = tarfh.extractfile(member)
            lines = fh.readlines()
            ligandname = lines[0].decode().strip().split()[1]
            if ligandname in params_kept: continue
            params_kept.add(ligandname)
            if ligandname in ligandnames:
                fh = tarfh.extractfile(member)
                outtarfh.addfile(member, fh)
                n += 1
            if n == len(ligandnames):
                break
    if n < len(ligandnames):
        print(f"Warning: only {n}/{len(ligandnames)} were found.")
    print("Saved: %s"%outtarparamsfn)
    outtarfh.close()

    #extract out files
    tarfn2prefix = {}
    lignames2prefix = {}
    for ligandname,description in lignames2descriptions.items():
        prefix = description.split(".")[0]
        lignames2prefix[ligandname] = prefix
        if not os.path.exists(tar_result_fn):
            raise IOError(f"{tar_result_fn} doesn't exits!")
        if tar_result_fn not in tarfn2prefix:
            tarfn2prefix[tar_result_fn] = [prefix]
        else:
            tarfn2prefix[tar_result_fn].append(prefix)
    prefix2silentfns = {}
    for tarfn in tarfn2prefix:
        retdict = extract_silentfns_from_tarfn(tarfn2prefix[tarfn], tarfn, temp_silentdir)
        prefix2silentfns.update(retdict)

    if not os.path.exists(outpdbpath):
        os.makedirs(outpdbpath)
    inargs = []
    rosettahome = os.environ.get('ROSETTAHOME')
    if rosettahome is None:
        raise Exception("Error: env variable ROSETTAHOME is not set.")
    print(f"ROSETTAHOME: {rosettahome}")
    rosettaapp = os.path.join(rosettahome,
                            "source/bin/extract_pdbs.linuxgccrelease")
    for ligandname,prefix in lignames2prefix.items():
        description = lignames2descriptions[ligandname]
        
        slientfn = prefix2silentfns[prefix]
        extract_pdb = os.path.expanduser(rosettaapp)
        cmd = [extract_pdb]
        cmd.extend(["-gen_potential", "-overwrite", "-beta_cart"])
        cmd.extend(["-in:file:extra_res_fa", outtarparamsfn])
        cmd.extend(["-in:file:silent",slientfn])
        cmd.extend(["-in:file:tags", description])
        cmd.extend(["-missing_density_to_jump"])
        cmd.extend(["-in:file:fullatom"])
        cmd.extend(["-out:prefix", ligandname+"."])
        print(" ".join(cmd))
        inargs.append((cmd, outpdbpath, logfn))
    nproc=16
    with mp.Pool(nproc) as pool:
        results = pool.starmap(run_cmd, inargs)
    print(results)

def save_top_N_pdbs_all(curr_iter, configfn, ntop, outdir):
    config = load_configfn(configfn)

    summary_path = config['result_path']
    tar_result_path = config['concat_tar_path']
    params_path = os.path.join(config['project_tempdir'], 'params')
    temprootdir= os.path.join(config['project_tempdir'], "extract")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tempdir = os.path.join(temprootdir, "iter_all")
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    df_all = None
    for i_iter in range(1, curr_iter+1):
        summary_fn = os.path.join(summary_path, f"{config['project_name']}_train{i_iter}_vs_results.aug.feather")
        if not os.path.exists(summary_fn):
            print(f"Cannot find {summary_fn}")
            continue
        df = pd.read_feather(summary_fn)
        df['i_iter'] = [i_iter]*len(df)
        if df_all is None:
            df_all = df
        else:
            df_all = df_all.append(df, ignore_index=True)
        
    df_all.sort_values(by='dG', inplace=True, ignore_index=True)
    print(df_all.head())
    iter2ligandnames = {}
    iter2descriptions = {}
    
    for irow in range(ntop):
        iter2ligandnames.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['ligandname'])
        iter2descriptions.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['description'])
    for i_iter in iter2ligandnames:
        tar_result_fn = os.path.join( tar_result_path, f"{config['project_name']}_train{i_iter}.tar" )
        params_fn = os.path.join(params_path, f"train{i_iter}_params.tar")
        ligandnames = iter2ligandnames[i_iter]
        descriptions = iter2descriptions[i_iter]
        extract_pdbs_from_tar_resultfn(tar_result_fn, params_fn, tempdir, outdir, ligandnames, descriptions)
    df_top = df_all[:ntop]
    summaryfn = os.path.join(outdir, "summary.feather")
    df_top.to_feather(summaryfn)
    summaryfn = os.path.join(outdir, "summary.csv")
    df_top.to_csv(summaryfn)

def save_top_N_pdbs(prefixes, configfn, ntop, outdir):
    config = load_configfn(configfn)

    summary_path = config['result_path']
    tar_result_path = config['concat_tar_path']
    params_path = os.path.join(config['project_tempdir'], 'params')
    temprootdir= os.path.join(config['project_tempdir'], "extract")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tempdir = os.path.join(temprootdir, prefixes[-1])
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    df_all = None
    for prefix in prefixes:
        summary_fn = os.path.join(summary_path, f"{config['project_name']}_{prefix}_vs_results.aug.feather")
        if not os.path.exists(summary_fn):
            print(f"Cannot find {summary_fn}")
            return
        df = pd.read_feather(summary_fn)
        df['i_iter'] = [prefix]*len(df)
        if df_all is None:
            df_all = df
        else:
            df_all = df_all.append(df, ignore_index=True)
        
    df_all.sort_values(by='dG', inplace=True, ignore_index=True)
    print(df_all.head())
    iter2ligandnames = {}
    iter2descriptions = {}

    for irow in range(ntop):
        iter2ligandnames.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['ligandname'])
        iter2descriptions.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['description'])
    for i_iter in iter2ligandnames:
        tar_result_fn = os.path.join( tar_result_path, f"{config['project_name']}_{i_iter}.tar" )
        params_fn = os.path.join(params_path, f"{i_iter}_params_0.tar")
        ligandnames = iter2ligandnames[i_iter]
        descriptions = iter2descriptions[i_iter]
        extract_pdbs_from_tar_resultfn(tar_result_fn, params_fn, tempdir, outdir, ligandnames, descriptions)
    df_top = df_all[:ntop]
    summaryfn = os.path.join(outdir, "summary.feather")
    df_top.to_feather(summaryfn)
    summaryfn = os.path.join(outdir, "summary.csv")
    df_top.to_csv(summaryfn)

def save_top_N_cluster_pdbs_all(clusterfn, configfn, ntop, outdir):
    config = load_configfn(configfn)

    summary_path = config['result_path']
    tar_result_path = config['result_path']
    params_path = os.path.join(config['project_tempdir'], 'params')
    temprootdir= os.path.join(config['project_tempdir'], "extract")

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    tempdir = os.path.join(temprootdir, "iter_all")
    if not os.path.exists(tempdir):
        os.makedirs(tempdir, exist_ok=True)

    df_all = pd.read_feather(clusterfn)
        
    df_all.sort_values(by='dG', inplace=True, ignore_index=True)
    df_all.drop_duplicates(subset=['clusterid'], keep='first',inplace=True, ignore_index=True)
    print(df_all.head())
    iter2ligandnames = {}
    iter2descriptions = {}
    
    for irow in range(ntop):
        iter2ligandnames.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['ligandname'])
        iter2descriptions.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['description'])
    for i_iter in iter2ligandnames:
        if os.path.exists(os.path.join( tar_result_path, f"{config['project_name']}_train{i_iter}.tar" )):
            tar_result_fn = os.path.join( tar_result_path, f"{config['project_name']}_train{i_iter}.tar" )
        elif os.path.exists(os.path.join( tar_result_path, f"{config['project_name']}_{i_iter}.tar" )):
            tar_result_fn =os.path.join( tar_result_path, f"{config['project_name']}_{i_iter}.tar" )
            
        if os.path.exists(os.path.join(params_path, f"train{i_iter}_params_0.tar")):
            params_fn = os.path.join(params_path, f"train{i_iter}_params_0.tar")
        elif os.path.exists(os.path.join(params_path, f"{i_iter}_params_0.tar")):
            params_fn = os.path.join(params_path, f"{i_iter}_params_0.tar")
            
        ligandnames = iter2ligandnames[i_iter]
        descriptions = iter2descriptions[i_iter]
        extract_pdbs_from_tar_resultfn(tar_result_fn, params_fn, tempdir, outdir, ligandnames, descriptions)
    df_top = df_all[:ntop]
    summaryfn = os.path.join(outdir, "summary.feather")
    df_top.to_feather(summaryfn)
    summaryfn = os.path.join(outdir, "summary.csv")
    df_top.to_csv(summaryfn)

def save_top_N_pdbs_pdfn(clusterfn, configfn, ntop, outdir, prefix=None):
    config = load_configfn(configfn)

    summary_path = config['result_path']
    tar_result_path = config['result_path']
    params_path = os.path.join(config['project_tempdir'], 'params')
    temprootdir= os.path.join(config['project_tempdir'], "extract")

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    tempdir = os.path.join(temprootdir, "iter_all")
    if not os.path.exists(tempdir):
        os.makedirs(tempdir, exist_ok=True)

    df_all = pd.read_feather(clusterfn)
        
    df_all.sort_values(by='dG', inplace=True, ignore_index=True)
    print(df_all.head())
    iter2ligandnames = {}
    iter2descriptions = {}
    if prefix and "i_iter" not in df_all:
        df_all["i_iter"] = [prefix]*len(df_all)
    ntop = min(ntop, len(df_all))
    for irow in range(ntop):
        iter2ligandnames.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['ligandname'])
        iter2descriptions.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['description'])
    for i_iter in iter2ligandnames:
        if os.path.exists(os.path.join( tar_result_path, f"{config['project_name']}_train{i_iter}.tar" )):
            tar_result_fn = os.path.join( tar_result_path, f"{config['project_name']}_train{i_iter}.tar" )
        elif os.path.exists(os.path.join( tar_result_path, f"{config['project_name']}_{i_iter}.tar" )):
            tar_result_fn =os.path.join( tar_result_path, f"{config['project_name']}_{i_iter}.tar" )
            
        if os.path.exists(os.path.join(params_path, f"train{i_iter}_params_0.tar")):
            params_fn = os.path.join(params_path, f"train{i_iter}_params_0.tar")
        elif os.path.exists(os.path.join(params_path, f"{i_iter}_params_0.tar")):
            params_fn = os.path.join(params_path, f"{i_iter}_params_0.tar")
            
        ligandnames = iter2ligandnames[i_iter]
        descriptions = iter2descriptions[i_iter]
        extract_pdbs_from_tar_resultfn(tar_result_fn, params_fn, tempdir, outdir, ligandnames, descriptions)
    df_top = df_all[:ntop]
    summaryfn = os.path.join(outdir, "summary.feather")
    df_top.to_feather(summaryfn)
    summaryfn = os.path.join(outdir, "summary.csv")
    df_top.to_csv(summaryfn)

    
def main_save_pdb():
    configfn = "../config_zinc22_db.json"
    i_iter = 11
    config = load_configfn(configfn)
    outrootdir = os.path.join(config['project_path'], "top_pdbs", f"iter{i_iter}")
    ntop = 100
    outdir = os.path.join(outrootdir, f"top{ntop}_pdbs_all")
    save_top_N_pdbs_all(i_iter, configfn, ntop=ntop, outdir=outdir)

def main_save_cluster_pdb():
    configfn = "../config_zinc22_db.json"
    prefix = "substructure"
    ntop = 100
    clusterfn = f"./cluster.{prefix}.300000.feather"
    config = load_configfn(configfn)
    outrootdir = os.path.join(config['project_path'], "top_cluster_pdbs", f"{prefix}")
    outdir = os.path.join(outrootdir, f"top{ntop}_pdbs_all")
    save_top_N_cluster_pdbs_all(clusterfn, configfn, ntop, outdir)

def main_save_pdbs_dbfn():
    configfn = "../config_zinc22_db.json"
    zincid = "Z3009405982"
    prefix = "substructure"
    ntop = 100
    dbfn = "similar_in_substructure.feather"
    config = load_configfn(configfn)
    outrootdir = os.path.join(config['project_path'], "top_similar_pdbs", f"{zincid}")
    outdir = os.path.join(outrootdir, f"top{ntop}_pdbs_all")
    save_top_N_pdbs_pdfn(dbfn, configfn, ntop, outdir, prefix)

if __name__ == "__main__":
    #main_save_pdb()
    #main_save_cluster_pdb()
    main_save_pdbs_dbfn()

