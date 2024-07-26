import os,sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from openvs.models import VanillaNet2
from openvs.args import VanillaModelArgs
from openvs.data import PredictDataSet
from openvs.utils import *

import pandas as pd
import orjson
from tap import Tap
from time import time

DTYPE=torch.float32

class PredictArgs(Tap):
    modelfn: str = None
    database_type: str = None
    database_path: str = None
    prediction_path: str = None
    disable_progress_bar: bool = True
    batch_size : int = 10000
    outfileformat: str = "feather"
    run_platform: str="auto" #Literal["gpu", "slurm", "auto"], I need "auto" to be default
    i_iter: int
    fplist: str

def get_args(i_iter, modelfn, config):
    predict_args = PredictArgs()
    predict_args.parse_args()

    predict_args.database_type = config["database_type"]
    predict_args.database_path = config["fps_path"]
    
    prediction_path = os.path.join( config["prediction_path"], f"model_{i_iter}_prediction")
    predict_args.prediction_path = prediction_path
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    
    predict_args.disable_progress_bar = True
    predict_args.batch_size = 50000
    predict_args.outfileformat = "feather"
    predict_args.modelfn = modelfn

    model_args = VanillaModelArgs()
    args_dict = {
        "dataset_type":"binaryclass"
    }
    model_args.from_dict(args_dict)

    return predict_args, model_args

def load_model(model, args: PredictArgs, model_args: VanillaModelArgs=None):
    modelfn = args.modelfn
    if model is not None:
        return model
    
    if model_args is None:
        raise Exception("Model is None and no Model Args found!")
    
    if args.run_platform == "gpu":
        if not torch.cuda.is_available():
            raise Exception("gpu is requested but gpu is not available!")
        device = torch.device("cuda")
    elif args.run_platform == "slurm":
        if torch.cuda.is_available():
            print("Warning: GPU is available but running on SLURM CPU is specified.")
        device = torch.device("cpu")
    elif args.run_platform == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VanillaNet2(model_args)
    model.to(device)
    print(f"Loaded model {modelfn}")
    if not torch.cuda.is_available():
        ckpt = torch.load(modelfn, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(modelfn)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Best model from epoch {ckpt['epoch']}")
    metrics = ["accuracys", "precisions", "recalls", "efs", "roc_auc"]
    for metric in  metrics:
        print(f"Previous best model test {metric}:", ckpt[f"test_{metric}"])
    for metric in  metrics:
        print(f"Previous best model validation {metric}:", ckpt[f"val_{metric}"])
    return model

def predict_dbfn(dbfn, outfn, model, args: PredictArgs, model_args: VanillaModelArgs=None):

    if os.path.exists(outfn):
        print(f"{outfn} exists.")
        return 0
    
    try:
        model = load_model(model, args, model_args)
        model.eval()
        model.to(DTYPE)

        predict_dataset = PredictDataSet(dbfn)
        if dbfn.endswith(".csv"):
            db_df = pd.read_csv(dbfn)
        elif dbfn.endswith(".feather"):
            db_df = pd.read_feather(dbfn)
        else:
            raise Exception(f"{dbfn} file format is wrong.")
        if args.batch_size >= len(predict_dataset):
            bs = len(predict_dataset)
        else:
            bs = args.batch_size
        predict_loader=DataLoader(predict_dataset, batch_size=bs, shuffle=False, drop_last=False)
        print(f"Predicting {dbfn}...")
        all_predicts = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for batch in tqdm(predict_loader, total=len(predict_loader), leave=False, disable=args.disable_progress_bar):
                x = batch['X'].to(DTYPE).to(device)
                y_hat = model(x).to(DTYPE)
                predicts = torch.squeeze(y_hat)
                if len(predicts.shape) == 0: # if y_hat is (1,1) predicts will be a scalar
                    predicts = torch.unsqueeze(predicts,dim=0)
                predicts = predicts.detach().cpu().numpy()
                all_predicts.extend( predicts )
        if "molecule_id" in db_df.columns:
            id_column = "molecule_id"
        elif "zinc_id" in db_df.columns:
            id_column = "zinc_id"
        elif "ZINCID" in db_df.columns:
            id_column = "ZINCID"
        elif "zincid" in db_df.columns:
            id_column = "zincid"
        else:
            id_column = "molecule_id"
        if 'relpath' in db_df:
            out_dicts = {"molecule_id": db_df[id_column], "p_hits": all_predicts, 
                        "smiles": db_df["smiles"], "relpath": db_df["relpath"]}
        else:
            out_dicts = {"molecule_id": db_df[id_column], "p_hits": all_predicts, "smiles": db_df["smiles"]}
        predicts_df = pd.DataFrame(out_dicts)
        if args.outfileformat == "csv":
            predicts_df.to_csv(outfn, index=False)
        elif args.outfileformat == "feather":
            predicts_df.to_feather(outfn)
        print(f"Saved: {outfn}")
    except Exception as e:
        print(e)
        return dbfn

def predict_dbfn_wrapper(inargs):
    dbfn, outfn, model, args, model_args = inargs
    return predict_dbfn(dbfn, outfn, model, args, model_args)


def predict_database(dblistfn, args: PredictArgs, model_args: VanillaModelArgs, config):
    database_path = args.database_path
    assert(database_path is not None)
    assert(args.prediction_path is not None)
    outrootdir = args.prediction_path
    if not os.path.exists(outrootdir):
        os.makedirs(outrootdir)

    print("Predict Args:", args)
    print("Model Args:", model_args)
    if args.run_platform == "gpu":
        if not torch.cuda.is_available():
            raise Exception("gpu is requested but gpu is not available!")
        device = torch.device("cuda")
    elif args.run_platform == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.run_platform == "cpu":
        device = torch.device('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = None
    model = load_model(model, args, model_args)
    model.to(DTYPE)
    model.eval()

    dbtype = args.database_type
    
    with open(dblistfn, 'r') as infh:
        dbfns = [l.strip() for l in infh]
    
    t0 = time()
    print("Predicting the database...")
    joblist = []
    print(f"Number of fingerprints files: {len(dbfns)}")
    for dbfn in dbfns:
        if not os.path.exists(dbfn):
            continue
        subdir = os.path.basename(os.path.dirname(dbfn))
        dbfname = os.path.basename(dbfn)
        outdir = os.path.join(outrootdir, subdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfname = ".".join(dbfname.split(".")[:-1] + ["predict", args.outfileformat])
        outfn = os.path.join(outdir, outfname)
        if os.path.exists(outfn):
            print(f"{outfn} exists.")
            continue

        predict_dbfn(dbfn, outfn, model, args)
    t1 = time()
    print(f"Inference took {t1-t0} s")

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

def predict_dbfns(dbfilelist, i_iter, configfn):
    config = load_configfn(configfn)
    model_path = config['model_path']
    modelfn = os.path.join(model_path, f"model_{i_iter}", "vanilla_model_best.pt")
    if not os.path.exists(modelfn):
        raise Exception(f"Cannot find {modelfn}")

    pargs, margs = get_args(i_iter, modelfn, config)
    #pargs.run_platform = 'slurm'
    print(f"Running platform: {pargs.run_platform}")
    predict_database(fpfilelist, pargs, margs, config)


if __name__ == "__main__":
    configfn = os.path.join("../", "config_zinc22_db.json" )
    args = PredictArgs().parse_args()
    fpfilelist = args.fplist
    predict_dbfns(fpfilelist, args.i_iter, configfn)
