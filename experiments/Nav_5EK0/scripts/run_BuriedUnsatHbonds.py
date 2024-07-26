import os,sys

from glob import glob
import subprocess as sp
import tarfile
import pandas as pd
import orjson
import multiprocessing as mp

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def run_cmd(cmd, wdir, logfn):
    os.makedirs(wdir, exist_ok=True)
    with open(logfn, 'w') as logfile:
        p = sp.Popen(cmd, cwd=wdir, stdout=logfile)
        p.communicate()
    return p.returncode

def extract_silentfns_from_tarfn(tags, tarfn, tempdir):
    prefix2silentfns = {}
    tags_set = set(tags)
    with tarfile.open(tarfn, 'r') as tarfh:
        for member in tarfh.getmembers():
            if not member.name.endswith("out"):
                continue
            prefix = os.path.basename(member.name).strip(".out")
            if prefix not in tags_set:
                continue
            outsilentfn = os.path.join(tempdir, f"{prefix}.out")
            # overwrite each time
            if os.path.exists(outsilentfn):
                cmd = f"rm {outsilentfn}"
                p = sp.Popen(cmd, shell=True)
                p.communicate()

            fh=tarfh.extractfile(member)
            content = fh.read()
            with open(outsilentfn, 'wb') as outfh:
                outfh.write(content)
            print(f"Wrote: {outsilentfn}")
            prefix2silentfns[prefix] = outsilentfn
            
            if len(prefix2silentfns) == len(tags):
                break

    return prefix2silentfns


def run_topN_BuriedUnsatHbonds(prefix, configfn, ntop, outdir):
    config = load_configfn(configfn)
    summary_path = config['result_path']
    tar_result_path = config['concat_tar_path']
    params_path = os.path.join(config['project_tempdir'], 'params')
    temprootdir= os.path.join(config['project_tempdir'], "extract")


    os.makedirs(outdir, exist_ok=True)
    tempdir = os.path.join(temprootdir, "iter_all")
    os.makedirs(tempdir, exist_ok=True)

    df_all = None
    summary_fn = os.path.join(summary_path, f"{config['project_name']}_{prefix}_vs_results.aug.feather")
    print(f"Reading summary file: {summary_fn}")
    if not os.path.exists(summary_fn):
        print(f"Cannot find {summary_fn}")
        return
    df_all = pd.read_feather(summary_fn)
    df_all['i_iter'] = [prefix]*len(df_all)
        
    df_all.sort_values(by='dG', inplace=True, ignore_index=True)
    print(df_all.head())
    iter2ligandnames = {}
    iter2descriptions = {}
    
    for irow in range(ntop):
        iter2ligandnames.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['ligandname'])
        iter2descriptions.setdefault( df_all.iloc[irow]["i_iter"], [] ).append(df_all.iloc[irow]['description'])
    for prefix in iter2ligandnames:
        tar_result_fn = os.path.join( tar_result_path, f"{config['project_name']}_{prefix}.tar" )
        params_fn = os.path.join(params_path,  f"{prefix}_params.tar")
        ligandnames = iter2ligandnames[prefix]
        descriptions = iter2descriptions[prefix]
        BuriedUnsatHbonds_helper(tar_result_fn, params_fn, tempdir, outdir, ligandnames, descriptions)

def run_topN_BuriedUnsatHbonds_all(curr_iter, configfn, ntop, outdir):
    config = load_configfn(configfn)

    summary_path = config['result_path']
    tar_result_path = config['result_path']
    params_path = os.path.join(config['project_tempdir'], 'params')
    temprootdir= os.path.join(config['project_tempdir'], "extract")
    projname = config['project_name']


    os.makedirs(outdir, exist_ok=True)

    tempdir = os.path.join(temprootdir, "iter_all")
    os.makedirs(tempdir, exist_ok=True)

    df_all = None
    for i_iter in range(1, curr_iter+1):
        summary_fn = os.path.join(summary_path, f"{projname}_train{i_iter}_vs_results.aug.feather")
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
        tar_result_fn = os.path.join( tar_result_path, f"{projname}_train{i_iter}.tar" )
        params_fn = os.path.join(params_path, f"train{i_iter}_params.tar")
        ligandnames = iter2ligandnames[i_iter]
        descriptions = iter2descriptions[i_iter]
        BuriedUnsatHbonds_helper(tar_result_fn, params_fn, tempdir, outdir, ligandnames, descriptions)

def BuriedUnsatHbonds_one_pdb(ligandname, pdbfn, paramsfn, outpath):
    os.makedirs(outpath, exist_ok=True)
    rosettahome = os.environ.get('ROSETTAHOME')
    if rosettahome is None:
        raise Exception("Error: env variable ROSETTAHOME is not set.")
    print(f"ROSETTAHOME: {rosettahome}")
    rosettaapp = os.path.join(rosettahome,
                            "source/bin/rosetta_scripts.linuxgccrelease")
    xmlfn=os.path.join(os.getcwd(), "buns.xml")
    app = os.path.expanduser(rosettaapp)
    logfile = f"{outpath}/{ligandname}.log"
    cmd = [app]
    cmd.extend(["-s", pdbfn])
    cmd.extend(["-in:file:extra_res_fa", paramsfn])
    cmd.extend(["-holes:dalphaball", os.path.join(rosettahome, "/source/external/DAlpahBall/DAlphaBall.gcc")])
    cmd.extend(["-gen_potential", "-overwrite", "-beta_cart"])
    cmd.extend(["-parser:protocol", xmlfn])
    cmd.extend(["-crystal_refine", "-renumber_pdb"])
    cmd.extend(["-out:path:pdb", outpath])
    cmd.extend(["-out:prefix", ligandname+"."])

    return run_cmd(cmd, outpath, logfile)


def BuriedUnsatHbonds_helper(tar_result_fn, tarparamsfn, tempdir, outpdbpath, ligandnames, descriptions):
    
    if len(ligandnames) != len(descriptions):
        raise 
    lignames2descriptions = dict(zip(ligandnames, descriptions))
    temp_paramsdir = os.path.join(tempdir, "params")
    temp_silentdir = os.path.join(tempdir, "silentfns")

    os.makedirs(temp_paramsdir, exist_ok=True)
    os.makedirs(temp_silentdir, exist_ok=True)

    #extract params files
    outtarparamsfn = os.path.join(temp_paramsdir, "ligand_params.tar")

    ligands_listfn = os.path.join(temp_paramsdir, "ligands_list.txt")
    with open(ligands_listfn, 'w') as outfh:
        content = [ligandname+"\n" for ligandname in ligandnames]
        outfh.writelines(content)
        print("Saved: %s"%ligands_listfn)
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
            if ligandname in ligandnames:
                fh = tarfh.extractfile(member)
                outtarfh.addfile(member, fh)
                paramfn = os.path.basename(member.name)
                n += 1
            if n == len(ligandnames):
                break
            
    print("Saved: %s"%outtarparamsfn)
    outtarfh.close()

    #extract out files
    silentfns = []
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

    os.makedirs(outpdbpath, exist_ok=True)
    joblist = []
    xmlfn=os.path.join(os.getcwd(), "buns.xml")
    rosettahome = os.environ.get('ROSETTAHOME')
    if rosettahome is None:
        raise Exception("Error: env variable ROSETTAHOME is not set.")
    print(f"ROSETTAHOME: {rosettahome}")
    rosettaapp = os.path.join(rosettahome,
                            "source/bin/rosetta_scripts.linuxgccrelease")
    for ligandname,prefix in lignames2prefix.items():
        description = lignames2descriptions[ligandname]
        logfile = f"{outpdbpath}/{ligandname}.log"
        slientfn = prefix2silentfns[prefix]
        extract_pdb = os.path.expanduser(rosettaapp)
        cmd = [extract_pdb]
        cmd.extend(["-holes:dalphaball", os.path.join(rosettahome, "source/external/DAlpahBall/DAlphaBall.gcc")])
        cmd.extend(["-gen_potential", "-overwrite", "-beta_cart"])
        cmd.extend(["-parser:protocol", xmlfn])
        cmd.extend(["-in:file:extra_res_fa", outtarparamsfn])
        cmd.extend(["-in:file:silent",slientfn])
        cmd.extend(["-in:file:tags", description])
        cmd.extend(["-crystal_refine", "-renumber_pdb"])
        cmd.extend(["-out:path:pdb", outpdbpath])
        cmd.extend(["-out:prefix", ligandname+"."])
        joblist.append((cmd, outpdbpath, logfile))
    ncpus=16
    print("Number of jobs:", len(joblist))
    with mp.Pool(ncpus) as pool:
        results = pool.starmap(run_cmd, joblist)
    print(results)

def finished(zincid, existed_pdbfns):
    for pdbfn in existed_pdbfns:
        if zincid in pdbfn:
            return True
    return False

def run_cluster_top():
    configfn = "../config_zinc22_db.json"
    config = load_configfn(configfn)
    ntop = 1000
    i_iter = 7
    prefix = f"train{i_iter}"
    clusterfn = f"cluster.iter{i_iter}.100000.feather"
    outdir = os.path.join(config['project_path'], "top_pdbs_unsat", prefix)
    df_all = pd.read_feather(clusterfn)
    df_all.sort_values(by='dG', inplace=True, ignore_index=True)
    print(df_all.head())

    params_path = os.path.join(config['project_tempdir'], 'top_params_cluster', f"iter_{i_iter}", f"params_top{ntop}")
    if not os.path.exists(params_path):
        raise Exception(f"{params_path} doesn't exist.")
 

    inpdbdir = os.path.join(config['project_path'], "top_cluster_pdbs", f"iter{i_iter}", f"top{ntop}_pdbs_all_holo")
    print(inpdbdir)
    if not os.path.exists(inpdbdir):
        raise Exception(f"{inpdbdir} doesn't exist.")
    patt = os.path.join(inpdbdir, "*.pdb")
    pdbfns = glob(patt)
    inargs = []
    
    patt = os.path.join(outdir, "*.pdb")
    existed_pdbfns = glob(patt)
    
    
    for pdbfn in pdbfns:
        fname = os.path.basename(pdbfn)
        zincid = fname.split(".")[0]
        
        if not zincid.startswith("ZINC"):
            raise Exception(f"{zincid} is not valide")
        if finished(zincid, existed_pdbfns): continue
        paramsfn = os.path.join(params_path, f"{zincid}.params")
        if not os.path.exists(paramsfn):
            print(f"Cannot find {paramsfn}")
            continue
        inarg = (zincid, pdbfn, paramsfn, outdir)
        inargs.append(inarg)
    ncpus = 16
    print(f"Number of mp jobs: {len(inargs)}")
    with mp.Pool(ncpus) as pool:
        retvals = pool.starmap(BuriedUnsatHbonds_one_pdb, inargs)
    print(retvals)


if __name__ == "__main__":
    run_cluster_top()
