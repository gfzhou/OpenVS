import os,sys
from glob import glob
from ccdc.io import MoleculeReader
from ccdc.conformer import GeometryAnalyser
import multiprocessing as mp
import pandas as pd
from distributed import Client
from dask_jobqueue import SLURMCluster
from pathlib import Path
import orjson

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def analyze_geometry(infn, outfn, out_torsionfn = None):
    engine = GeometryAnalyser()
    mol_reader = MoleculeReader(infn)
    mol = mol_reader[0]
    mol_reader.close()
    mol.assign_bond_types(which='unknown')
    mol.standardise_aromatic_bonds()
    mol.standardise_delocalised_bonds()
    mol.add_hydrogens()
    geometry_analysed_mol = engine.analyse_molecule(mol)
    n_torsions = len([t for t in geometry_analysed_mol.analysed_torsions if t.unusual])
    if out_torsionfn is not None:
        content = []
        for torsion in geometry_analysed_mol.analysed_torsions:
            if torsion.unusual and torsion.enough_hits:
                tor_atoms = ', '.join(label for label in torsion.atom_labels)
                content.append(f"{tor_atoms}\n")
        with open(out_torsionfn, 'w') as outf:
            outf.writelines(content)
        print(f"Saved: {out_torsionfn}")

    filename = Path(os.path.basename(infn)).with_suffix('')
    entry = [filename, n_torsions]
    print(filename, n_torsions)
    df = pd.DataFrame([entry], columns=['ligandname', 'n_bad_torsions'] )
    df.to_csv(outfn)
    print(f"Saved: {outfn}")

def analyze_geometry_wrapper(inargs):
    return analyze_geometry(*inargs)

def analyze_top_pdbs(configfn, mode="slurm"):
    config = load_configfn(configfn)
    i_iter=8
    inpath = f"../top_cluster_pdbs/iter{i_iter}/top1000_pdbs_all_ligand"
    outpath = f"../top_cluster_pdbs/iter{i_iter}/top1000_ligand_geometry_analysis"
    os.makedirs(outpath, exist_ok=True)
    pattern = os.path.join(inpath, "*.pdb")
    infns = sorted(glob(pattern))

    if mode == "slurm":
        cluster_obj = SLURMCluster(cores=1, processes=1, memory="5GB",
                        queue='short', job_name="geo_analyser",
                        job_script_prologue=['source ~/.bashrc', 'conda activate csd'],
                        walltime="3:00:00")
        ncpus = min(300, len(infns))
        cluster_obj.adapt(minimum=0, maximum=ncpus, wait_count=400)
        client = Client(cluster_obj)
        print(f"Using slurm clusters, {ncpus} cores:")
        print(client.scheduler_info())
    joblist = []
    for i, infn in enumerate(infns):
        filename = Path(os.path.basename(infn)).with_suffix('')
        outfn = os.path.join(outpath, f"{filename}.csv")
        out_torsionfn = os.path.join(outpath, f"{filename}.bad_torsions")
        if os.path.exists(outfn) and os.path.exists(out_torsionfn): continue
        inargs = (infn, outfn, out_torsionfn)
        if mode == "slurm":
            joblist.append(client.submit(analyze_geometry_wrapper, inargs))
        elif mode == "mp":
            joblist.append(inargs)
        else:
            print(f"Working on {i+1}/{len(infns)}")
            analyze_geometry_wrapper(inargs)
    
    if mode == "slurm":
        print("Number slurm jobs:", len(joblist))
        client.gather(joblist)
    elif mode == "mp":
        print("Number mutiprocessing jobs:", len(joblist))
        ncpu = 16
        with mp.Pool(ncpu) as pool:
            dfs_raw = pool.map(analyze_geometry_wrapper, joblist)
        print("Done with multiprocessing jobs.")

def main(configfn):
    analyze_top_pdbs(configfn, mode='mp')

if __name__ == "__main__":
    configfn = "../config_zinc22_db.json"
    main(configfn)
