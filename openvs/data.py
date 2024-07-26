import os,sys
from tkinter import W
import pandas as pd
import numpy as np
from typing import Any, Callable, List, Tuple, Union
from typing_extensions import Literal
from tap import Tap 
import subprocess as sp

from rdkit import DataStructs

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from openvs.utils.utils import *

class ECFPDataSet(Dataset):
    
    def __init__(self, infns, nBits=1024, score_column="dG", cutoff=None, drop_duplicate_column=None):
        self.nBits = nBits
        if isinstance(infns, str):
            print(f"Loading {infns}")
            if infns.endswith(".csv"):
                self.raw_data = pd.read_csv(infns)
            elif infns.endswith(".feather"):
                self.raw_data = pd.read_feather(infns)
            else:
                raise Exception(f"{infns} file format is wrong.")
        elif isinstance(infns, list):
            self.raw_data = []
            if drop_duplicate_column is not None:
                columns = [drop_duplicate_column, 'fp_binary', score_column]
            else:
                columns = ['fp_binary', score_column] #TODO: check the validity
                
            for infn in infns:
                print(f"Loading {infn}")
                if infn.endswith(".csv"):
                    self.raw_data.append( pd.read_csv(infn)[columns] )
                elif infn.endswith(".feather"):
                    self.raw_data.append( pd.read_feather(infn)[columns] )
                else:
                    raise Exception(f"{infn} file format is wrong.")
            self.raw_data = pd.concat(self.raw_data, ignore_index=True)

        if drop_duplicate_column is not None:
            # if we drop duplicates, we want to keep the best score ones.
            self.raw_data.sort_values(by=score_column, inplace=True)
            print(f"Sorted data by {score_column}")
            print(self.raw_data.head()[score_column])
            print(f"Drop duplicates in {drop_duplicate_column}")
            print("Original number of data:", len(self.raw_data))
            self.raw_data.drop_duplicates(subset=drop_duplicate_column, keep="first", inplace=True, ignore_index=True)
            print("After removing dupcliates, N_data:", len(self.raw_data))

        self.bitarrays = np.zeros((len(self.raw_data), nBits), dtype=np.int8)
        self.scores = np.zeros(len(self.raw_data))
        self.cutoff = cutoff
        self.pos = 0 # less than cutoff, number of positive case
        self.neg = 0 # larger than cutoff, number of negative case
        self.score_column = score_column
        self.setup()
        print(f"Done loading {infns}, Ndata: {len(self.raw_data)}")

    def setup(self):
        if "fp_binary" in self.raw_data:
            for i, fp2 in enumerate(self.raw_data["fp_binary"]):
                fp = DataStructs.ExplicitBitVect(fp2)
                self.bitarrays[i,:] = fp.ToList()[:]
        elif "fp_hexstring" in self.raw_data:
            print("Will be deprecated in the future, use binary fingerprints")
            fp_bitstrings = self.raw_data.apply(lambda row: format(int(row.fp_hexstring, 16), '0%db'%self.nBits), axis=1 )
            self.bitarrays[:] = np.stack( fp_bitstrings.apply( lambda x: np.array(list(map(int,x)), dtype=int) ), axis=0 )[:]
        elif "smiles" in self.raw_data:
            print("Warning: generating fingerprints from smiles on the fly!")
            self.bitarrays[:] = smiles_to_bitarrays(list(self.raw_data['smiles']), nBits=self.nBits)[:]
        else:
            raise Exception("Cannot find binary fp, hexstrings or smiles")
        self.set_cutoff(self.cutoff)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"X":self.bitarrays[idx], "Y":self.scores[idx]}
        return sample
    
    def set_cutoff(self, cutoff):
        if self.cutoff is None:
            self.scores[:] = list( map(float,  self.raw_data[self.score_column]) )
        else:
            self.scores[:] = list( map(int, self.raw_data[self.score_column]<cutoff) )
            self.pos = sum(self.scores)
            self.neg = len(self.raw_data) - self.pos

    def print_status(self):
        if self.cutoff is None:
            print("Cutoff not specified")
            return
        print("N<cutoff: %d, N>cutoff: %d, Ntotal: %d"%(self.pos, self.neg, len(self.raw_data)))

class DockingDataSet(Dataset):
    def __init__(self, csvfn, nBits=1024, cutoff=None, score_column="dG"):
        self.raw_data = pd.read_csv(csvfn)
        self.fp_bitarray = np.zeros((len(self.raw_data), nBits),dtype=np.int8)
        self.scores = np.zeros(len(self.raw_data))
        self.cutoff = cutoff
        self.pos = 0 # less than cutoff, positive case
        self.neg = 0 # larger than cutoff, negative case
        for i in range(len(self.raw_data)):
            fp_hexstring = self.raw_data["fp_hexstring"][i]
            fp_bitstring = format(int(fp_hexstring,16), '0%db'%nBits)
            self.fp_bitarray[i] = np.array(list(map(int,fp_bitstring)),dtype=int)
            if cutoff is None:
                self.scores[i] = float( self.raw_data[score_column][i] )
            else:
                self.scores[i] = int( self.raw_data[score_column][i]<cutoff )
        
        if cutoff is not None:
            self.pos = sum(self.scores)
            self.neg = len(self.raw_data) - self.pos

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"X":self.fp_bitarray[idx], "Y":self.scores[idx]}
        return sample

    def print_status(self):
        if self.cutoff is None:
            print("Cutoff not specified")
            return
        print("N<cutoff: %d, N>cutoff: %d"%(self.pos, self.neg))

class PredictDataSet(Dataset):
    def __init__(self, infn, nBits=1024):
        if infn.endswith(".csv"):
            self.raw_data = pd.read_csv(infn)
        elif infn.endswith(".feather"):
            self.raw_data = pd.read_feather(infn)
        else:
            raise Exception(f"{infn} file format is wrong.")

        self.bitarrays = np.zeros((len(self.raw_data), nBits), dtype=np.int8)
        status = False
        if "fp_binary" in self.raw_data:
            try:
                print("Loading binary fingerprints")
                for i, fp2 in enumerate(self.raw_data["fp_binary"]):
                    fp = DataStructs.ExplicitBitVect(fp2)
                    self.bitarrays[i,:] = fp.ToList()[:]
                status = True
            except:
                print("Failed to load binary fingerprints")
                
        if not status and "fp_hexstring" in self.raw_data:
            print("Will be deprecated in the future, use binary fingerprints")
            fp_bitstrings = self.raw_data.apply(lambda row: format(int(row.fp_hexstring, 16), '0%db'%nBits), axis=1 )
            self.bitarrays[:] = np.stack( fp_bitstrings.apply( lambda x: np.array(list(map(int,x)), dtype=int) ), axis=0 )[:]
            
        if not status and "smiles" in self.raw_data:
            print("Warning: generating fingerprints from smiles on the fly!")
            self.bitarrays[:] = smiles_to_bitarrays(list(self.raw_data['smiles']), nBits=nBits)[:]

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"X":self.bitarrays[idx]}
        return sample



if __name__ == "__main__":
    pass
