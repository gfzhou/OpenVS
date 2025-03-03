import os,sys
import numpy as np
import torch
from time import time

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
            self.one_to_all_d = one_to_all_gpu_euclidean

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
