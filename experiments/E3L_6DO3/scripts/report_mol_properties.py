import os,sys
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Crippen, QED, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def add_mol_properties(indf):
    assert 'smiles' in indf.columns, "Cannot find smiles column."
    qed, clogP, sa = [], [], []
    for smi in indf['smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            qed.append(np.nan)
            clogP.append(np.nan)
            sa.append(np.nan)
        qed.append( QED.qed(mol) )
        clogP.append( Crippen.MolLogP(mol) )
        sa.append( sascorer.calculateScore(mol) )
    df_new = indf.copy()
    df_new['QED'] = qed
    df_new['clogP'] = clogP
    df_new['SA'] = sa
    return df_new
    
def report_molecule_properties(infn, outfn):
    if infn.endswith(".csv"):
        indf = pd.read_csv(infn)
    elif infn.endswith(".feather")
        indf = pd.read_feather(infn)
    else:
        raise ValueError("Input file format not supported.")
    
    df_new = add_mol_properties(indf)
    
    if outfn.endswith(".csv"):
        df_new.to_csv(outfn, index=False)
    elif outfn.endswith(".feather"):
        df_new.to_feather(outfn)
    else:
        raise ValueError("Output file format not supported.")

