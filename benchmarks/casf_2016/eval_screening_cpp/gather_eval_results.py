import os,sys
import multiprocessing as mp
import subprocess as sp
from glob import glob
import multiprocessing as mp

def run_cmd(cmd, wdir):
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, cwd=wdir)
    p.communicate()
    return p.returncode

def gather_receptor(results_basedir, rec, outfn):
    
    dGfn = os.path.join(results_basedir, f"{rec}_dG.txt")
    if not os.path.exists(dGfn):
        cmd = f"grep dG {rec}/????/*.pdb > {rec}_dG.txt"
        print(cmd)
        run_cmd(cmd, results_basedir)
    
    content = ["#code_ligand_num    score\n"]
    with open(dGfn, 'r') as infh:
        for l in infh:
            fields = l.strip().split()
            dG = float(fields[-1])
            code = fields[0].split("/")[-1].split(".")[1]
            code = code.replace("complex_", "")
            code = code.replace("_0001", "")
            content.append(f"{code} {dG}\n")

    with open(outfn, 'w') as outfh:
        outfh.writelines(content)
    print(f"Saved: {outfn}")

def main(overwrite=False):
    rectrglistfn = "trglist.screening.txt"
    with open(rectrglistfn, 'r') as infh:
        reclist = [l.strip() for l in infh]
    outdir = "./results_screening_relax_lig_simple_no_cst"
    os.makedirs(outdir, exist_ok=True)
    arglist = []
    results_basedir = "eval_screening_cpp_relax_lig_simple_no_cst"
    for rec in reclist:
        outfn = os.path.join(outdir, f"{rec}_score.dat")
        if os.path.exists(outfn) and not overwrite: continue
        inargs = (results_basedir, rec, outfn)
        arglist.append(inargs)

    nprocs = 16
    with mp.Pool(nprocs) as pool:
        pool.starmap(gather_receptor, arglist)

if __name__ == '__main__':
    main()

