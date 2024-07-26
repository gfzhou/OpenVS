import os,sys
from openvs.utils.utils import load_configfn
from openvs.utils.params_utils import gen_params_from_folder


def gen_params(i_iter, configfn, runmode="multiprocessing", chargemodel='zinc', multimol2=False, infer_atomtypes=False):
    config = load_configfn(configfn)
    indir = os.path.join(config['mol2_path'], f"{chargemodel}_mol2s_iter{i_iter}")
    if not os.path.exists(indir):
        raise Exception(f"{indir} doesn't exist.")
    outdir = os.path.join(config['project_tempdir'], "params", f"train{i_iter}_params" )
    gen_params_from_folder(indir, outdir, mode=runmode, 
                            overwrite=False, nopdb=True, 
                            mol2gen_app=None, multimol2=multimol2, 
                            infer_atomtypes=infer_atomtypes,
                            queue='cpu')


if __name__ == "__main__":
    chargemodel = 'zinc22'
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    configfn = os.path.join("../", "config_zinc22_db.json")
    infer_atomtypes = True
    multimol2 = False
    runmode = "mp"
    gen_params(i_iter, configfn, runmode, chargemodel, multimol2, infer_atomtypes)
