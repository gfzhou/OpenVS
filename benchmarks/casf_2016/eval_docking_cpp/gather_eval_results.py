import os,sys
import multiprocessing as mp
import subprocess as sp
from glob import glob
import multiprocessing as mp

def run_cmd(cmd, wdir):
    p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, cwd=wdir)
    p.communicate()
    return p.returncode

def gather_target(trg, resultsdir, outfn, score_tag="dG"):
    
    resultdir_trg = os.path.join(resultsdir, trg)
    pdbfns = sorted(glob(os.path.join(resultdir_trg, "*.pdb")))
    if score_tag == "dG":
        identifier = "dH"
    elif score_tag == "score":
        identifier = "pose"
    content = ["#code    score\n"]
    for pdbfn in pdbfns:
        with open(pdbfn, 'r') as infh:
            for l in infh:
                if not l.startswith(identifier): continue
                fields = l.strip().split()
                score = float(fields[-1])
                break
        code = pdbfn.split(".")[1]
        fields = code.split("_")
        decoyid = ""
        if len(fields) == 4:
            trg = fields[1]
            decoyid = fields[2]
        elif len(fields) == 3:
            trg = fields[0]
            decoyid = "ligand"
        assert decoyid != ""
        code = f"{trg}_{decoyid}"
        content.append(f"{code} {score}\n")

    with open(outfn, 'w') as outfh:
        outfh.writelines(content)
    print(f"Saved: {outfn}")

def main(overwrite=False):
    rectrglistfn = "trglist.core.txt"
    results_basedir = os.path.join(os.getcwd(), "eval_docking_cpp_relax_lig_mcentropy_ligcst1")
    with open(rectrglistfn, 'r') as infh:
        reclist = [l.strip() for l in infh]
    score_tag = "score"
    outdir = f"./gathered_results/eval_docking_cpp_{score_tag}"
    os.makedirs(outdir, exist_ok=True)
    arglist = []
    for trg in reclist:
        outfn = os.path.join(outdir, f"{trg}_score.dat")
        if os.path.exists(outfn) and not overwrite: continue
        inargs = (trg, results_basedir, outfn, score_tag)
        arglist.append(inargs)
    nprocs = 16
    with mp.Pool(nprocs) as pool:
        pool.starmap(gather_target, arglist)

if __name__ == '__main__':
    main()

