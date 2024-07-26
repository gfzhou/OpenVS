from __future__ import print_function
import os,sys
import numpy as np
import subprocess as sp
import gzip
from glob import glob
import pandas as pd
from pathlib import Path


ROCbins = [0.005,0.01,0.02,0.05,0.1]

def run_cmd(cmd):
    p = sp.Popen(cmd, shell=True) 
    p.communicate()

def buns_penalty(nbuns):
    penalty = 0.0
    if nbuns>2 and nbuns<=4:
        penalty = np.max([0, nbuns-2])*0.5
    if nbuns>4:
        penalty = np.max([0, nbuns-4])*2.0
    return penalty


def _check_tags_ndx_line(line, tags):

    indices = []
    valid_tags = []

    fields = line.strip().split()
    #print(fields)
    for tag in tags:
        try:
            if tag == "description":
                indices.append(-1)
                valid_tags.append(tag)
            else:
                indices.append(fields.index(tag))
                valid_tags.append(tag)
        except ValueError:
            pass
            #print( "Cannot find tag:%s in file, skip it."%tag )

    return valid_tags,indices

def _line_parser(line,indices=None,dtype=float):
    values = []
    if indices is None:
        return line
    else:
        fields = line.strip().split()
        for i in indices:
            try:
                values.append(dtype(fields[i]))
            except (ValueError,IndexError):
                try:
                    values.append(fields[i])
                except IndexError:
                    pass
        if len(values) != len(indices):
            return False
        return values

def read_log_fstream(infh, tags, line_marker = "SCORE:" , tags_marker = "description", ignore_incomplete=False):
    valid_tags = []
    values = []
    for l in infh:
        try:
            line = l.decode()
        except AttributeError:
            line = l
        if (line_marker == "nomarker") and (tags_marker not in line): #read every line except with the tag marker
            pass
        elif (not line.startswith(line_marker)):
            continue
        if tags_marker in line:
            valid_tags, indices = _check_tags_ndx_line(line, tags)
            continue
        if ignore_incomplete and len(valid_tags) < len(tags):
            continue
        reval = _line_parser(line,indices,dtype=float)
        if reval:
            values.append(reval)
    infh.close()

    return valid_tags, values

def read_log(fn, tags, line_marker = "SCORE:" , tags_marker = "description", ignore_incomplete=False):

    if not os.path.exists(fn):
        raise IOError("%s doesn't exist!"%fn)

    values = []
    if ".gz" in fn:
        infn = gzip.open(fn, 'rt')
    else:
        infn = open(fn, 'r')
    return read_log_fstream(infn, tags, line_marker, tags_marker, ignore_incomplete)


def auc(sortable): #(dG,cat)
    if isinstance(sortable, list):
        sortable.sort()
    if isinstance(sortable, np.ndarray):
        sortable = sortable[sortable[:,0].argsort()]
    
    nTtot = 0
    nFtot = 0
    for dG,cat in sortable:
        if cat == 0: nFtot += 1
        else: nTtot += 1
    
    nTot = len(sortable) 
    fTot = (nTtot*1.0/nTot)
    fTF = (nTtot*1.0/nFtot)
    
    fT = 0.0
    nT = 0
    nF = 0
    fT_as_fF = []

    nbin    = [int(f*nTot) for f in ROCbins ]
    nbinROC = [max(1,int(f*nFtot)) for f in ROCbins]
    enrich = [0.0 for k in nbin]
    enrichROC = [0.0 for k in nbin]
    for i,(dE,cat) in enumerate(sortable):
        if cat == 1:
            fT += 1.0/nTtot
            nT += 1
        else:
            fT_as_fF.append(fT)
            nF += 1
        n = nT+nF
        
        if n in nbin:
            ibin = nbin.index(n)
            enrich[ibin] = (nT*1.0/n)/fTot
        if nF in nbinROC:
            ibin = nbinROC.index(nF)
            enrichROC[ibin] = (nT*1.0/nF)/fTF

    return (sum(fT_as_fF)/len(fT_as_fF)), enrichROC

def gather_results_siletnfns_helper(silentfns, tags):
    all_values = []
    for silentfn in silentfns:
        valid_tags, values = read_log(silentfn, tags)
        if len(tags) != len(valid_tags):
            print(f"Valid tags: {valid_tags} doesn't match specified tags {tags}!")
            return None
        all_values.extend(values)
    df = pd.DataFrame(all_values, columns=valid_tags)
    return df

def grep_dG_trg_silent(trg, rundir, prefix="", score_tag='dG'):
    ligands_fn = os.path.join(Path(__file__).parents[1], "lists", "%s_ligands.txt"%trg)
    decoys_fn = os.path.join(Path(__file__).parents[1], "lists", "%s_decoys.txt"%trg)
    with open(ligands_fn, 'r') as infn:
        ligands = set([ l.strip() for l in infn])
    with open(decoys_fn, 'r') as infn:
        decoys = set([l.strip() for l in infn])

    dGs = []
    patt = os.path.join(rundir, f"*.{prefix}out")
    infns = glob(patt)
    if score_tag == 'dG':
        tags = [ "dG", "ligandname"]

    if len(infns) == 0:
        return []
    df = gather_results_siletnfns_helper(infns, tags)
    if df is None:
        raise Exception("Failed collecting results!")

    for i in range(len(df)):
        dG = float( df['dG'][i] )
        ligname = df['ligandname'][i]
        if ligname in ligands:
            dGs.append((dG,1))
        elif ligname in decoys:
            dGs.append((dG,0))
            
    return dGs

def run_trgs_simple(trgs):
    aucs = []
    enrichs = []
    print("%-12s %-14s %-14s | ER "%("#Target","AUC", "N")+" %5.4f"*len(ROCbins)%tuple(ROCbins) )
    datadir = "./outputs/results_vsx/"
    print(os.path.abspath(datadir))
    score_tag = 'dG'
    for trg in trgs:
        if not os.path.exists(os.path.join(datadir, "%s_results"%trg)):
            continue
        aucs_runs = []
        enrichs_runs = []
        ndGs = 0
        for i in range(3):
            rundir = os.path.join(datadir, f"{trg}_results", f"run{i}")
            dGs = grep_dG_trg_silent(trg, rundir, "%d."%i, score_tag=score_tag)
            dGs = np.array(dGs)

            if len(dGs) == 0: continue
            
            ndGs += len(dGs)

            val = auc(dGs)
            if not val: continue

            aucs_runs.append(val[0])
            enrichs_runs.append(val[1])

        ndGs = ndGs // 3
        if len(aucs_runs) == 0: continue
        enrich = np.array(enrichs_runs).mean(axis=0)

        if len(enrichs) ==0 :
            enrichs = list(enrich)
        else:
            enrichs = [enrich[i]+a for i,a in enumerate(enrichs)]
        aucs.append(np.mean(aucs_runs))

        print( "%-12s %6.4f %.3f %d |   "%(trg, np.mean(aucs_runs), np.std(aucs_runs), ndGs)+" %5.1f"*len(enrich)%tuple(enrich) )

    enrichs = [a/len(aucs) for a in enrichs]
    print( "AVRG:        %6.4f |   "%(sum(aucs)/len(aucs))+" %5.1f"*len(enrichs)%tuple(enrichs))

if __name__ == "__main__":
    with open("../lists/trglist.txt", 'r') as infh:
        trgs = [l.strip() for l in infh]
    trgs = ["ada"]
    run_trgs_simple(trgs)
