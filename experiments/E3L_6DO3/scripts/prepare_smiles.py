import os,sys
import orjson
import numpy as np
import subprocess as sp
import pandas as pd
from glob import glob
from distributed import Client
from dask_jobqueue import SLURMCluster


def run_cmd(cmd):
    p = sp.Popen(cmd, shell=True)
    p.communicate()
    return p.returncode

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config


def extract_smiles_from_db_file(dbfn, outdir, prefix,  molid_header="molecule_id", totalN=500000, batchsize=2000, overwrite=False):
    df = pd.read_feather(dbfn)[:totalN]
    assert "smiles" in df, f"Cannot find smiles in {dbfn}"
    assert molid_header in df, f"Cannot find {molid_header} in {dbfn}"

    contents = []
    smifns = []
    counter = 0
    os.makedirs(outdir, exist_ok=True)
    cmd = f"rm {outdir}/{prefix}*.smi"
    p = sp.Popen(cmd, shell=True)
    p.communicate()
    for i in range(len(df)):
        smi = df['smiles'][i]
        molid = df[molid_header][i]
        # skip invalid molids
        if not ( molid.startswith('Z') or molid.startswith('PV')):
            continue
        smi = f'{smi} {molid}\n'
        contents.append(smi)
        if len(contents) == batchsize:
            outfn_ndx = counter//batchsize
            smifn = os.path.join(outdir, f"{prefix}.{outfn_ndx}.smi")
            with open(smifn, "w") as outf:
                outf.writelines(contents)
            contents = []
            print("wrote: %s"%smifn)
            smifns.append(smifn)
        counter += 1

    if len(contents) != 0:
        outfn_ndx = counter//batchsize
        smifn = os.path.join(outdir, f"{prefix}.{outfn_ndx}.smi")
        with open(smifn, "w") as outf:
            outf.writelines(contents)
        contents = []
        print("wrote: %s"%smifn)
        smifns.append(smifn)

    return smifns

def protonate_smiles(inargs):
    smifn, outdir = inargs
    smifname = os.path.basename(smifn)
    protname = smifname.replace(".smi", ".prot.smi")
    os.makedirs(outdir, exist_ok=True)
    outfn = os.path.join(outdir, protname)
    dimorphite_home = os.environ.get('DIMORPHITE')
    app = f"{dimorphite_home}/dimorphite_dl.py"
    if not os.path.exists(app):
        raise ValueError(f"Cannot find {app}, please set the correct path.")
    cmd = f"python {app} --smiles_file {smifn} --min_ph 7.4 --max_ph 7.4 --output_file {outfn} --pka_precision 0.1"
    return run_cmd(cmd)

def protonate_smiles_dir(smi_dir, outdir, mode='slurm'):
    
    pattern = os.path.join(smi_dir, "*.smi")
    smifns = sorted(glob(pattern))
    os.makedirs(outdir, exist_ok=True)
    cmd = f"rm {outdir}/*.smi"
    p = sp.Popen(cmd, shell=True)
    p.communicate()

    if mode == 'slurm':
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="1GB",
                queue='cpu', job_name="protsmi_worker",
                walltime="3:00:00")
        ncpus = max(1, min(100, int(len(smifns)//10)))
        cluster_obj.adapt(minimum=0, maximum=ncpus, wait_count=400)
        client = Client(cluster_obj)
        print("Using slurm clusters, N_cpus", ncpus)
        print(client.scheduler_info())
    joblist = []
    for smifn in smifns:
        inargs = (smifn, outdir)
        if mode == 'slurm':
            joblist.append(client.submit(protonate_smiles, inargs))
        else:
            protonate_smiles(inargs)
    
    if mode == 'slurm':
        print("Number jobs:", len(joblist))
        print( client.gather(joblist) )
        client.close()

def save_failed_smiles(rawdir, protdir, outdir, batchsize=2000):
    pattern = os.path.join(rawdir, "*.smi")
    raw_smifns = sorted(glob(pattern))
    contents = []
    os.makedirs(outdir, exist_ok=True)
    cmd = f"rm {outdir}/*.smi"
    p = sp.Popen(cmd, shell=True)
    p.communicate()
    for raw_smifn in raw_smifns:
        smifname = os.path.basename(raw_smifn).replace(".smi", ".prot.smi")
        smifn_prot = os.path.join(protdir, smifname)
        with open(smifn_prot, 'r') as infh:
            prot_molids = [l.strip().split()[1] for l in infh]
            prot_molids = set(prot_molids)
        with open(raw_smifn, 'r') as infh:
            for l in infh:
                smi, molid = l.strip().split()
                if molid not in prot_molids:
                    contents.append(l)
    if len(contents) == 0:
        return    
    begin_idx = np.arange(0, len(contents), batchsize)
    for i, idx in enumerate(begin_idx):
        smifn = os.path.join(outdir, f"obabel.{i}.smi")
        with open(smifn, "w") as outf:
            outf.writelines(contents[idx:idx+batchsize])
        print("wrote: %s"%smifn)


def prepare_smiles_real_db(i_iter, configfn, totalN, batchsize=2000, debug=False, mode='slurm'):
    config = load_configfn(configfn)
    topfn = os.path.join(config['prediction_path'], f"model_{i_iter}_prediction", "top_predictions", "all.top.feather")
    assert os.path.exists(topfn), f"Cannot find {topfn}"
    molid_header="molecule_id"
    prefix = "gen3d_top"
    raw_outdir = os.path.join(f"smiles_iter{i_iter+1}", "raw")
    os.makedirs(raw_outdir, exist_ok=True)
    if debug:
        raw_outdir = os.path.join("debug_gen3d", "raw")
        os.makedirs(raw_outdir, exist_ok=True)
    extract_smiles_from_db_file(topfn, raw_outdir, prefix, molid_header, totalN, batchsize)
    prefix = "gen3d_random"
    randomfn = os.path.join(config['prediction_path'], f"model_{i_iter}_prediction", "top_predictions", "all.random.feather")
    extract_smiles_from_db_file(randomfn, raw_outdir, prefix, molid_header, totalN, batchsize)

    print("Protonate smiles...")
    prot_outdir = os.path.join(f"smiles_iter{i_iter+1}", "prot") 
    os.makedirs(prot_outdir, exist_ok=True)
    protonate_smiles_dir(raw_outdir, prot_outdir, mode)
    print("Done")

    print("Save non-protonated cases..")
    obabel_outdir = os.path.join(f"smiles_iter{i_iter+1}", "obabel")
    os.makedirs(obabel_outdir, exist_ok=True)
    save_failed_smiles(raw_outdir, prot_outdir, obabel_outdir, batchsize)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    configfn = os.path.join("../", "config_real_db.json" )
    debug=False
    overwrite=False
    totalN = 10
    batchsize = 10
    mode = 'mp'
    prepare_smiles_real_db(i_iter, configfn, totalN, batchsize, debug, mode)

