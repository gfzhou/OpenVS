'''Usefual functions for smi file handling,fingerprint calculation and statistical analysis.'''

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

from collections.abc import Iterable
from typing import Any, Callable, List, Sequence, Tuple, Union
from typing_extensions import Literal

import orjson

from dask import compute, delayed
import dask.multiprocessing

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss, precision_score, recall_score

def load_smiles_file(input_file: str, delimiter: str = " ") -> pd.DataFrame:
    '''Load smiles infomation from csv-like smi file.

    Params
    ======

    - input_file: Path of input smi file.
    - delimiter = ' ': Delimiter for csv-like file.

    Returns
    =======

    DataFrame including ss,ids and other descriptions.
    '''
    data = pd.read_csv(input_file, delimiter=delimiter)
    return data

def smi2fp_bitstring_helper(smi: str, morgan_radius: int =2, nBits: int = 1024) -> str:
    '''Convert smiles string into morgan fingerprint(0/1 format string).
    
    Params
    ======
    - smi: A smiles string.
    - morgan_radius = 2: Morgan radius of substructure to calculate fingerprints,need to be int.
    - nBits = 1024: Length of the fingerprint vector.

    Returns
    =======
    Binary format string of the fingerprint.
    '''
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=nBits)
    fp_bitstring = fp.ToBitString()
    return fp_bitstring

def smi2fp_hexstring_helper(smi:str, morgan_radius:int =2, nBits:int=1024) -> bytes:
    '''Convert smile string into morgan fingerprint's bytes(0x hexadecimal format).
    
    Params
    ======
    - smi: A smile string.
    - morgan_radius = 2: Morgan radius of substructure to calculate fingerprints,need to be int.
    - nBits = 1024: Length of the fingerprint vector.

    Returns
    =======
    bytes of the fingerprint of the smile string.
    '''
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=nBits)
    fp_bitstring = fp.ToBitString()
    fp_hexstring = format(int(fp_bitstring, 2), 'x')
    return fp_hexstring

def smiles_to_bitstrings(smiles_list:Iterable[str], morgan_radius: int = 2, nBits: int = 1024) -> tuple[str]:
    """Convert a list of smiles strings into bitstrings in parallel.
    
    Params
    ======
    - smiles_list: A list of smiles strings.
    - morgan_radius = 2: Morgan radius of substructure,2 is roughly equivalent to ECFP 4
    - nBits = 1024: Length of the fingerprint vector.

    Returns
    =======

    A tuple of calculated morgan fingerprings(0/1-strings).
    """
    bitstrings = []
    jobs = []
    for i, smi in enumerate(smiles_list):
        jobs.append(delayed(smi2fp_bitstring_helper)(smi, morgan_radius, nBits))

    bitstrings = compute(*jobs, scheduler="processes")
    return bitstrings

def smiles_to_hexstrings(smiles_list:Iterable[str], morgan_radius:int =2, nBits:int=1024) -> tuple[bytes]:
    """Convert a list of smiles strings into hex bytes in parallel.
    
    Params
    ======
    - smiles_list: A list of smiles strings.
    - morgan_radius = 2: Morgan radius of substructure,2 is roughly equivalent to ECFP 4
    - nBits = 1024: Length of the fingerprint vector.

    Returns
    =======

    A tuple of calculated morgan fingerprings(hex-bytes).
    """
    bitstrings = []
    jobs = []
    for i, smi in enumerate(smiles_list):
        jobs.append(delayed(smi2fp_hexstring_helper)(smi, morgan_radius, nBits))

    bitstrings = compute(*jobs, scheduler="processes")
    return bitstrings

def smiles_to_hexstrings_slow(smiles_list:Iterable[str], morgan_radius:int =2, nBits:int=1024) -> list[bytes]:
    """Convert a list of smiles strings into hex bytes in order.
    
    Params
    ======
    - smiles_list: A list of smiles strings.
    - morgan_radius = 2: Morgan radius of substructure,2 is roughly equivalent to ECFP 4
    - nBits = 1024: Length of the fingerprint vector.

    Returns
    =======
    A list of calculated morgan fingerprings(hex-bytes).
    """
    bitstrings:list[bytes] = []
    for i, smi in enumerate(smiles_list):
        bitstrings.append( smi2fp_hexstring_helper(smi, morgan_radius, nBits) )

    return bitstrings

def smiles_to_bitarrays(smiles_list: Iterable[str], radius:int=2, nBits:int=1024, useFeature:bool=True, useChirality:bool=True) -> np.ndarray:
    """Convert a list of smiles strings into an int matrix.
    
    Params
    ======
    - smiles_list: A group of smiles strings.
    - radius = 2: Morgan radius of substructure,2 is roughly equivalent to ECFP 4.
    - nBits = 1024: Length of the fingerprint vector.
    - useFeature = True: If False,use ConnectivityMorgan; if True: use FeatureMorgan.
    - useChirality = True: If True, chirality information will be included as a part of the bond invariants.
    
    Returns
    =======
    An int numpy matrix(n_smi*nBits),each column is a morgen 0/1 vector fingerprint.
    """
    bitarrays = np.zeros((len(smiles_list), nBits), dtype=int)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=useFeature, useChirality=useChirality)
        bitarrays[i,:] = fp.ToList()[:]

    return bitarrays


def smiles_to_bitarrays_slow(smiles_list:Iterable[str], morgan_radius:int =2, nBits:int=1024):
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

def smiles_to_binary_fingerprints(smis:Iterable[str], radius:int=2, nBits:int=1024, useFeature:bool=True, useChirality:bool=True) -> list[bytes]:
    """Convert a list of smiles strings into an list of hex-format morgan fingerprints.
    
    Params
    ======
    - smiles_list: A group of smiles strings.
    - radius = 2: Morgan radius of substructure,2 is roughly equivalent to ECFP 4.
    - nBits = 1024: Length of the fingerprint vector.
    - useFeature = True: If False,use ConnectivityMorgan; if True: use FeatureMorgan.
    - useChirality = True: If True, chirality information will be included as a part of the bond invariants.
    
    Returns
    =======
    An list of morgan fingerprints,each item is a morgen hex-format fingerprint.
    """
    fps:list[bytes] = []
    for i, smi in enumerate(smis):
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits, useFeatures=useFeature, useChirality=useChirality)
        fps.append(fp.ToBinary())
    return fps

def get_accuracys(targets: Sequence[int], preds: Union[List[float], List[List[float]]], thresholds: Iterable[float] = (0.2, 0.35, 0.5) ) -> np.ndarray:
    """Calculate accuracy according to a group of thresholds.
    
    Params
    ======
    - targets: 0/1 sample values
    - preds: Float prediction vector(s).If a prediction > threshold,it will be 1;otherwise 0.
    - thresholds: A group of thresholds for calculating accuracy respectively.
    
    Returns
    =======
    A numpy n_threshold * n_preds ndarray,each line is accuracy score between targets and predictions.
    """
    acc = []
    for threshold in thresholds:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction
        acc.append(accuracy_score(targets, hard_preds))
    return np.array(acc)

def get_precisions(targets: Sequence[int],
                    preds: Union[List[float],
                    List[List[float]]],
                    thresholds: Iterable[float]=(0.2, 0.35, 0.5)) -> np.ndarray:
    """Calculate precision according to a group of thresholds.
    
    Params
    ======
    - targets: 0/1 sample values
    - preds: Float prediction vector(s).If a prediction > threshold,it will be 1;otherwise 0.
    - thresholds: A group of thresholds for calculating precision respectively.
    
    Returns
    =======
    A numpy n_threshold * n_preds ndarray,each line is precision score between targets and predictions.
    """
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

def get_recalls(targets: Sequence[int],
                preds: Union[List[float], List[List[float]]],
                thresholds: Iterable[float]=(0.2, 0.35, 0.5)) -> np.ndarray:
    """Calculate recalls according to a group of thresholds.
    
    Params
    ======
    - targets: 0/1 sample values
    - preds: Float prediction vector(s).If a prediction > threshold,it will be 1;otherwise 0.
    - thresholds: A group of thresholds for calculating recalls respectively.
    
    Returns
    =======
    A numpy n_threshold * n_preds ndarray,each line is recall score between targets and predictions.
    """
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

def get_enrichment_factors(random_p: float, topNs: Iterable[int], preds: Sequence[float] , truth: Sequence[float]) -> np.ndarray:
    """Calculate enrichment factors of top N.
    
    Params
    ======
    - random_p: 
    - topNs: A group of integers to calculate topN enrichment factors. 
    - preds: Float prediction vector.
    - truth: Sample vector.
    - thresholds: A group of thresholds for calculating recalls respectively.
    
    Returns
    =======
    A numpy n_topNs ndarray,each line is enrichment factor between predictions and samples.
    """
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
    I = np.argsort(pred_mean) #ascending order
    return np.array(pred_std)[I][:N]


def load_configfn(configfn: str) -> dict:
    '''Load a binary config file and convert into a config dict.
    
    Params
    ======
    configfn: Config file name.

    Returns
    =======
    A config dict.
    '''
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config



if __name__  == "__main__":
    pass
