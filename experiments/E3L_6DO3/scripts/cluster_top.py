import os,sys

import torch
import numpy as np
import pandas as pd
import orjson
from time import time

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def one_to_all_tanimoto(x, X):
    c = torch.sum(X*x, dim=1)
    a = torch.sum(X,dim=1)
    b = torch.sum(x)
    
    return 1-c.type(torch.float)/(a+b-c).type(torch.float)
    

def one_to_all_euclidean(x, X, dist_metric="euclidean"):
    return torch.sqrt(torch.sum((X-x)**2,dim=1))


class BestFirstClustering():
    def __init__(self, cutoff, dist_metric="tanimoto", dtype=torch.uint8):

        if dist_metric == "euclidean":
            self.cutoff = cutoff
            self.one_to_all_d = one_to_all_euclidean

        elif dist_metric == 'tanimoto':
            self.cutoff = cutoff
            self.one_to_all_d = one_to_all_tanimoto
        if torch.cuda.is_available():
            self.use_gpu = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist_metric=dist_metric
        self.centroids_ = None
        self.centroids_index_ = []
        self.assignments_ = []
        self.clusterid_ = 0
        self.dtype = dtype
        print(f"BestFirstClustering: Use GPU: {self.use_gpu}")

    def partial_fit_assign(self, X, index_outfn=None, assign_outfn = None):
        
        t1 = time()
        index_ = []
        assignments = -torch.ones(len(X),requires_grad=False).to(self.device)
        indices = torch.arange(0, len(X),requires_grad=False).to(self.device)
        clusterid = 0
        while True:
            mask = assignments==-1
            if len(X[mask])==0: break
            distances = self.one_to_all_d(X[mask][0], X[mask])
            dist_indices = indices[mask][distances <= self.cutoff]
            assignments[dist_indices] = clusterid
            clusterid += 1
        t2 = time()
        print(f"Clustering took {t2-t1} s")
        self.assignments_ = assignments.cpu()

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def smi_to_fp(smi):
    nBits = 1024
    radius = 2
    useFeature = False
    useChirality = False
    m = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, 
                            nBits=nBits, useFeatures=useFeature, useChirality=useChirality)
    return fp

def get_top_all_iter(i_iter, ntop, config):
    prefixes = [ f"train{i}" for i in range(1, i_iter+1)]
    return get_top_iter(prefixes, ntop, config)


def get_top_iter(prefixes, ntop, config):
    projname = config['project_name']
    datadir = config['database_path']
    df_all = None
    for prefix in prefixes:
        infn = os.path.join(datadir, f"{projname}_{prefix}_vs_results.aug.feather")
        df = pd.read_feather(infn)
        df['i_iter'] = [prefix]*len(df)
        if df_all is None:
            df.sort_values(by='dG', inplace=True)
            df_all = df[:ntop]
        else:
            df.sort_values(by='dG', inplace=True)
            df_all = pd.concat([df_all, df[:ntop]], ignore_index=True)
    df_all.sort_values(by='dG', inplace=True)
    return df_all[:ntop]

def cluster():
    i_iter=11
    ntop = 300000
    configfn = "../config_zinc22_db.json"
    config = load_configfn(configfn)
    df = get_top_all_iter(i_iter, ntop, config)
    df.reset_index(drop=True, inplace=True)
    print(df.head())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    nBits = 1024
    fps = [ DataStructs.ExplicitBitVect(fp) for fp in list( df['fp_binary'] ) ]
    bitarrays = np.zeros((len(fps), nBits), dtype=np.int8)
    for i in range(len(fps)):
        bitarrays[i,:] = fps[i].ToList()[:]
    print(len(fps))
    cutoff = 0.5
    
    t1 = time()
    X = torch.tensor(bitarrays, requires_grad=False, dtype=torch.uint8).to(device).view(-1,nBits)
    clustering = BestFirstClustering(cutoff=cutoff, dist_metric="tanimoto")
    clustering.partial_fit_assign(X)
    t2 = time()
    print(f"clustering took {t2-t1} s")
    print( "n clusters:", len(np.unique(clustering.assignments_)) )
    
    df['clusterid'] = list(clustering.assignments_.numpy().astype(int))
    
    outfn = f"cluster.iter{i_iter}.{ntop}.feather"
    df.to_feather(outfn)
    print(f"Saved: {outfn}")

def cluster_substructure_iter():
    prefix='substructure'
    ntop = 300000
    configfn = "../config_zinc22_db.json"
    config = load_configfn(configfn)
    df = get_top_iter([prefix], ntop, config)
    df.reset_index(drop=True, inplace=True)
    print(df.head())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    nBits = 1024
    fps = [ DataStructs.ExplicitBitVect(fp) for fp in list( df['fp_binary'] ) ]
    bitarrays = np.zeros((len(fps), nBits), dtype=np.int8)
    for i in range(len(fps)):
        bitarrays[i,:] = fps[i].ToList()[:]
    print(len(fps))
    cutoff = 0.4
    
    t1 = time()
    X = torch.tensor(bitarrays, requires_grad=False, dtype=torch.uint8).to(device).view(-1,nBits)
    clustering = BestFirstClustering(cutoff=cutoff, dist_metric="tanimoto")
    clustering.partial_fit_assign(X)
    t2 = time()
    print(f"clustering took {t2-t1} s")
    print( "n clusters:", len(np.unique(clustering.assignments_)) )
    
    df['clusterid'] = list(clustering.assignments_.numpy().astype(int))
    
    outfn = f"cluster.{prefix}.{ntop}.feather"
    df.to_feather(outfn)
    print(f"Saved: {outfn}")

    

def main():
    cluster()
    #cluster_substructure_iter()


if __name__ == '__main__':
    main()
    
