import os,sys
from time import time
# import scientific py
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import base64
# rdkit stuff
from rdkit import Chem
from rdkit.Chem import AllChem

from typing import Any, Callable, List, Tuple, Union
from typing_extensions import Literal

import orjson

from dask import compute, delayed
import dask.multiprocessing

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, precision_score, recall_score

def load_smiles_file(input_file, delimiter=" "):
    data = pd.read_csv(input_file, delimiter=delimiter)
    return data

def smi2fp_bitstring_helper(smi, morgan_radius =2, nBits=1024):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=nBits)
    fp_bitstring = fp.ToBitString()
    return fp_bitstring

def smi2fp_hexstring_helper(smi, morgan_radius =2, nBits=1024):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=nBits)
    fp_bitstring = fp.ToBitString()
    fp_hexstring = format(int(fp_bitstring, 2), 'x')
    return fp_hexstring

def smiles_to_bitstrings(smiles_list, morgan_radius =2, nBits=1024):
    """
    morgan_radius = 2 is roughly equivalent to ECFP 4
    """
    bitstrings = []
    jobs = []
    for i, smi in enumerate(smiles_list):
        jobs.append(delayed(smi2fp_bitstring_helper)(smi, morgan_radius, nBits))
    
    bitstrings = compute(*jobs, scheduler="processes")
    return bitstrings

def smiles_to_hexstrings(smiles_list, morgan_radius =2, nBits=1024):
    """
    morgan_radius = 2 is roughly equivalent to ECFP 4
    """
    bitstrings = []
    jobs = []
    for i, smi in enumerate(smiles_list):
        jobs.append(delayed(smi2fp_hexstring_helper)(smi, morgan_radius, nBits))
    
    bitstrings = compute(*jobs, scheduler="processes")
    return bitstrings

def smiles_to_hexstrings_slow(smiles_list, morgan_radius =2, nBits=1024):
    """
    morgan_radius = 2 is roughly equivalent to ECFP 4
    """
    bitstrings = []
    for i, smi in enumerate(smiles_list):
        bitstrings.append( smi2fp_hexstring_helper(smi, morgan_radius, nBits) )

    return bitstrings

def smiles_to_bitarrays(smiles_list, radius=2, nBits=1024, useFeature=True, useChirality=True):
    """
    morgan_radius = 2 is roughly equivalent to ECFP 4
    """
    bitarrays = np.zeros((len(smiles_list), nBits), dtype=int)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=useFeature, useChirality=useChirality)
        bitarrays[i,:] = fp.ToList()[:]

    return bitarrays


def smiles_to_bitarrays_slow(smiles_list, morgan_radius =2, nBits=1024):
    """
    morgan_radius = 2 is roughly equivalent to ECFP 4
    """
    fps_bitarray = np.zeros((len(smiles_list), nBits), dtype=int)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=nBits)
        fp_bitstring = fp.ToBitString()
        fp_hexstring = format(int(fp_bitstring, 2), 'x')
        fp_bitstring2 = format(int(fp_hexstring,16), '0%db'%nBits)

        assert(fp_bitstring == fp_bitstring2)
        fp_bitarray = np.array(list(map(int,fp_bitstring)),dtype=int)
        fps_bitarray[i] = fp_bitarray

    return fps_bitarray

def smiles_to_binary_fingerprints(smis, radius=2, nBits=1024, useFeature=True, useChirality=True):
    fps = []
    for i, smi in enumerate(smis):
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=useFeature, useChirality=useChirality)
        fps.append(fp.ToBinary())
    return fps

def get_accuracys(targets: List[int], preds: Union[List[float], List[List[float]]], thresholds: float = [0.2, 0.35, 0.5] ) -> float:
    acc = []
    for threshold in thresholds:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction
        acc.append(accuracy_score(targets, hard_preds))
    return np.array(acc)

def get_precisions(targets: List[int], 
                    preds: Union[List[float], 
                    List[List[float]]], 
                    thresholds: List[float]=[0.2, 0.35, 0.5]) -> List[float]:
    precisions = []
    for threshold in thresholds:
        hard_preds = [1 if p > threshold else 0 for p in preds]
        precisions.append(precision_score(targets, hard_preds))
    return np.array(precisions)

def get_FPDE(ys, ys_pred, threshold, random_p):
    ys_pred = np.array(ys_pred)
    I = np.argsort(ys_pred)[::-1] #descending order
    sorted_truth = np.array(ys)[I]
    N = len( np.where( ys_pred>=threshold )[0] )
    TP_topN = sum(sorted_truth[:N])
    TP_randomN = N*random_p
    fpde = TP_topN/TP_randomN
    return fpde

def get_recalls(targets: List[int], 
                preds: Union[List[float], List[List[float]]], 
                thresholds: List[float]=[0.2, 0.35, 0.5]) -> List[float]:
    recalls = []
    for threshold in thresholds:
        hard_preds = [1 if p > threshold else 0 for p in preds]
        recalls.append(recall_score(targets, hard_preds))
    return np.array(recalls)

def get_data_kept_percentage(preds, thresholds):
    preds = np.array(preds)
    ntotal = len(preds)
    retval = []
    for t in thresholds:
        retval.append(np.sum(preds>=t)/ntotal)
    return retval

def get_enrichment_factors(random_p, topNs, preds, truth):
    I = np.argsort(preds)[::-1] #descending order
    sorted_truth = np.array(truth)[I]
    EFs = []
    for N in topNs:
        N = int(N)
        if N == 0: N = 1
        TP_topN = sum(sorted_truth[:N])
        TP_randomN = N*random_p
        EFs.append(TP_topN/TP_randomN)
    return np.array(EFs)

def get_enrichment_factors2(random_p, topNs, preds, truth):
    I = np.argsort(preds)[::-1] #descending order
    sorted_truth = np.array(truth)[I]
    sorted_pres = np.array(preds)[I]
    EFs = []
    thresholds = []
    for N in topNs:
        N = int(N)
        if N == 0: N = 1
        TP_topN = sum(sorted_truth[:N])
        TP_randomN = N*random_p
        EFs.append(TP_topN/TP_randomN)
        thresholds.append(sorted_pres[N])
    return np.array(EFs), np.array(thresholds)

def recall_threshold_detector(target_recall: float, 
                                targets: List[int], preds: Union[List[float], 
                                List[List[float]]], eps: float = 1E-6) -> float:
    def biserch(target_value, xl, xr, vl=None, vr=None):
        
        if vl is None:
            vl = get_recalls(targets, preds, [xl] )[0]
        if vr is None:
            vr = get_recalls(targets, preds, [xr] )[0]

        if abs(xl-xr) <= eps:
            return xr, vr
        if abs(vl-target_value) <= eps:
            return xl, vl
        elif abs(vr-target_value) <= eps:
            return xr, vr
        else:
            xm = (xl+xr)/2.
            vm = get_recalls(targets,preds,[xm])[0]
            if abs(vm-target_value) <= eps:
                return xm, vm
            if target_value > vm:
                return biserch(target_value, xl, xm, vl, vm)
            else:
                return biserch(target_value, xm, xr, vm, vr)

    
    threshold, recall = biserch(target_recall, 0.0, 1.0)
    recall2 = get_recalls(targets, preds, [threshold])[0]
    print(threshold, recall, recall2)
    return threshold, recall2

def get_top_std(pred_mean, pred_std, N=100):
    I = np.argsort(pred_mean)[::-1] #descending order
    return np.array(pred_std)[I][:N]

def get_bottom_std(pred_mean, pred_std, N=100):
    I = np.argsort(pred_mean) #descending order
    return np.array(pred_std)[I][:N]

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config



if __name__  == "__main__":
    pass
