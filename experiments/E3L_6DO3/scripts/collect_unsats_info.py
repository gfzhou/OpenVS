import os,sys
import subprocess as sp
import pandas as pd


def collect_unsats_info():
    ntop=1000
    prefix = f"substructure_top{ntop}"
    outfn = f"../top_pdbs_unsat/{prefix}.unsats.txt"
    indir = f"../top_pdbs_unsat/{prefix}"
    cmd = f"grep -H 'unsatisfied hbonds' {indir}/*.log > {outfn}"
    p = sp.Popen(cmd, shell=True)
    p.communicate()
    zincids = []
    n_unsats = []
    with open(outfn, 'r') as infh:
        for l in infh:
            fields = l.strip().split(':')
            zincid = fields[0].split('/')[-1].split('.')[0]
            n_unsat = int( fields[-1].split()[-1] )
            zincids.append(zincid)
            n_unsats.append(n_unsat)
    
    df = pd.DataFrame( list(zip(zincids, n_unsats)), columns=['zincid', 'unsats'])
    outfn = f"../top_pdbs_unsat/{prefix}.unsats.csv"
    df.to_csv(outfn)
    print(f"Saved: {outfn}")
    
    df_less1 = df[df['unsats']<=1]
    df_less1.reset_index(drop=True, inplace=True)
    outfn = f"../top_pdbs_unsat/{prefix}.unsats.less1.csv"
    df_less1.to_csv(outfn)
    print(f"Saved: {outfn}")
    
    df_less1 = df[df['unsats']<=0]
    df_less1.reset_index(drop=True, inplace=True)
    outfn = f"../top_pdbs_unsat/{prefix}.unsats.less0.csv"
    df_less1.to_csv(outfn)
    print(f"Saved: {outfn}") 

if __name__ == '__main__':
    collect_unsats_info()
            