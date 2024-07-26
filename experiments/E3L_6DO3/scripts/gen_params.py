import os,sys
from openvs.utils.utils import load_configfn
from openvs.utils.params_utils import gen_params_from_folder


def gen_params_iteration(i_iter, configfn, chargemodel='mmff94', multimol2=False, infer_atomtypes=False):
    config = load_configfn(configfn)
    indir = os.path.join(config['mol2_path'], f"{chargemodel}_mol2s_iter{i_iter}")
    if not os.path.exists(indir):
        raise Exception(f"{indir} doesn't exist.")
    outdir = os.path.join(config['project_tempdir'], "params", f"train{i_iter}_params" )
    gen_params_from_folder(indir, outdir, mode='slurm', 
                            overwrite=False, nopdb=True, 
                            mol2gen_app=None, multimol2=multimol2, 
                            infer_atomtypes=infer_atomtypes,
                            queue='cpu')

def main_iteration(configfn):
    chargemodel = 'mmff94'
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    
    infer_atomtypes = True
    multimol2 = True
    gen_params_iteration(i_iter, configfn, chargemodel, multimol2, infer_atomtypes)

def main_substructure(configfn):
    molid = "Z3009405982"
    config = load_configfn(configfn)
    indir = os.path.join(config['mol2_path'], "substructure_mol2s", molid)
    if not os.path.exists(indir):
        raise Exception(f"{indir} doesn't exist.")
    outdir = os.path.join(config['project_tempdir'], "params", "substructure_params", molid )
    infer_atomtypes = True
    multimol2 = False
    gen_params_from_folder(indir, outdir, mode='slurm', 
                            overwrite=False, nopdb=True, 
                            mol2gen_app=None, multimol2=multimol2, 
                            infer_atomtypes=infer_atomtypes,
                            queue='cpu')

if __name__ == "__main__":
    configfn = os.path.join("../", "config_real_db.json")
    #main_substructure(configfn)
    main_iteration(configfn)
