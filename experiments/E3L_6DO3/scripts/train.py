import os,sys
import pandas as pd
import numpy as np
from tqdm import tqdm

import orjson

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from openvs.models import VanillaNet
from openvs.args import TrainArgs, VanillaModelArgs
from openvs.data import ECFPDataSet
from openvs.utils.utils import *

from sklearn.metrics import roc_auc_score

EnableBar = True
if len(sys.argv) < 3:
    EnableBar = True
elif sys.argv[2].lower() == "false":
    EnableBar = False
print(f"Enable progress bar set to {EnableBar}")

DTYPE=torch.float32

def train_vanilla(args: TrainArgs, 
                  model_args: VanillaModelArgs,
                  writer: SummaryWriter = None) -> str:
    assert(args.model_path is not None)
    model_path= args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_dict_best = None
    modelfn_best = os.path.join(model_path, "vanilla_model_best.pt")
    if os.path.exists(modelfn_best):
        return modelfn_best
    
    disable_progress_bar = args.disable_progress_bar

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VanillaNet(model_args)
    model.to(device)
    print(model)

    bs = args.batch_size
    print_step = args.log_frequency

    train_datafn = args.train_datafn
    test_datafn = args.test_datafn
    validate_datafn = args.validate_datafn

    cutoff =args.score_cutoff
    print("Loading train/test/validation ...")
    if args.dataset_type == "binaryclass":
        train_set = ECFPDataSet(train_datafn, cutoff=cutoff, score_column=args.score_column, drop_duplicate_column="ligandname")
        test_set = ECFPDataSet(test_datafn, cutoff=cutoff, score_column=args.score_column, drop_duplicate_column="ligandname")
        val_set = ECFPDataSet(validate_datafn, cutoff=cutoff, score_column=args.score_column, drop_duplicate_column="ligandname")
    
    train_set.print_status()
    test_set.print_status()
    val_set.print_status()

    if args.weight_class:
        class_sample_count = np.array(
            [len(np.where(train_set.scores == t)[0]) for t in np.unique(train_set.scores)])
        weight = 1. / class_sample_count
        weight[1] *= 1.0
        samples_weight = np.array([weight[int(t)] for t in train_set.scores])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        samples_weight = np.ones(len(train_set.scores))
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    n_active_test = test_set.pos
    n_test = len(test_set)
    n_active_val = val_set.pos
    n_val = len(val_set)
    random_active_prob_test = float(n_active_test)/n_test
    random_active_prob_val = float(n_active_val)/n_val

    trainloader=DataLoader(train_set,  batch_size=bs, shuffle=False, drop_last=False, sampler=sampler)
    testloader=DataLoader(test_set,  batch_size=bs, shuffle=False, drop_last=False)
    valloader=DataLoader(val_set,  batch_size=bs, shuffle=False, drop_last=False)
    print("Done with loading.")

    if args.dataset_type == "binaryclass":
        criterion = nn.BCELoss()
    elif args.dataset_type == "regression":
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    print("Start training:")

    patience = args.patience
    min_test_loss=9999
    n_unchanged_test_loss=0
    best_epoch = 0
    topNs = [10, 100, 1000, 10000]
    thresholds = [0.2, 0.35, 0.5]
    model.to(DTYPE)
    for epoch in tqdm(range(args.epochs), disable=disable_progress_bar):
        running_loss = 0.0
        model.train()
        loss_train_epoch = []
        for sample_batch in tqdm(trainloader, total=len(trainloader), leave=False, disable=disable_progress_bar):
            model.zero_grad()
            x = sample_batch['X'].to(DTYPE).to(device)
            y = sample_batch['Y'].to(DTYPE).view(1,-1).to(device)
            y_hat = model(x).view(1,-1).to(DTYPE)
            train_loss = criterion(y_hat, y)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
            loss_train_epoch.append(train_loss.item())

        print("[%d] mean batch train_loss: %.6f"%(epoch, np.mean(loss_train_epoch)))
        if writer is not None:
            writer.add_scalar("train_loss", np.mean(loss_train_epoch), epoch)
        #------test------
        predict_test_running = []
        truth_test_running = []
        running_loss = 0.0
        model.eval()
        loss_test_epoch = []
        for sample_batch in tqdm(testloader, total=len(testloader), leave=False, disable=disable_progress_bar):
            x = sample_batch['X'].to(DTYPE).to(device)
            y = sample_batch['Y'].view(1,-1).to(DTYPE).to(device)
            y_hat = model(x).view(1,-1).to(DTYPE)
            for val in torch.squeeze(y).detach().cpu().numpy():
                truth_test_running.append(val)
            for val in torch.squeeze(y_hat).detach().cpu().numpy():
                predict_test_running.append(val)
            test_loss = criterion(y_hat, y)
            running_loss += test_loss.item()
            loss_test_epoch.append(test_loss.item())
 
        
        test_accuracys = get_accuracys(truth_test_running, predict_test_running, thresholds)
        test_precisions = get_precisions(truth_test_running, predict_test_running, thresholds)
        test_recalls = get_recalls(truth_test_running, predict_test_running, thresholds)
        test_auc = roc_auc_score(truth_test_running, predict_test_running)
        test_efs = get_enrichment_factors(random_active_prob_test, topNs, predict_test_running, truth_test_running)
        ave_test_loss = np.mean(loss_test_epoch)

        if writer is not None:
            writer.add_scalar("test_loss", ave_test_loss, epoch)
            writer.add_scalar("test_roc_auc", test_auc, epoch)
            writer.add_histogram("test_accuracys", test_accuracys, epoch )
            writer.add_histogram("test_precisions", test_precisions, epoch )
            writer.add_histogram("test_recalls", test_recalls, epoch )
            writer.add_histogram("test_EFs", test_efs, epoch )

        print(f"[{epoch}] test_loss: {ave_test_loss:.6f}," +
                " accuracys: "+ " ".join(f"{x:.3f}" for x in test_accuracys ) + 
                ", precision: "+ " ".join(f"{x:.3f}" for x in test_precisions) + 
                ", recall: " + " ".join(f"{x:.3f}" for x in test_recalls ) +
                f", roc_auc: {test_auc:.3f},"+
                " EFs: " + " ".join(f"{x:.3f}" for x in test_efs ))

        #------validate------
        predict_val_running = []
        truth_val_running = []
        running_loss = 0.0
        model.eval()
        loss_val_epoch = []
        for sample_batch in tqdm(valloader, total=len(valloader), leave=False, disable=disable_progress_bar):
            x = sample_batch['X'].to(DTYPE).to(device)
            y = sample_batch['Y'].view(1,-1).to(DTYPE).to(device)
            y_hat = model(x).view(1,-1).to(DTYPE)
            for val in torch.squeeze(y).detach().cpu().numpy():
                truth_val_running.append(val)
            for val in torch.squeeze(y_hat).detach().cpu().numpy():
                predict_val_running.append(val)
            val_loss = criterion(y_hat, y)
            running_loss += val_loss.item()
            loss_val_epoch.append(val_loss.item())
 
        val_accuracys = get_accuracys(truth_val_running, predict_val_running, thresholds)
        val_precisions = get_precisions(truth_val_running, predict_val_running, thresholds)
        val_recalls = get_recalls(truth_val_running, predict_val_running, thresholds)
        val_auc = roc_auc_score(truth_val_running, predict_val_running)
        
        val_efs = get_enrichment_factors(random_active_prob_val, topNs, predict_val_running, truth_val_running)
        ave_val_loss = np.mean(loss_val_epoch)

        if writer is not None:
            writer.add_scalar("val_loss", ave_val_loss, epoch)
            writer.add_scalar("val_roc_auc", val_auc, epoch)
            writer.add_histogram("val_accuracys", val_accuracys, epoch )
            writer.add_histogram("val_precisions", val_precisions, epoch )
            writer.add_histogram("val_recalls", val_recalls, epoch )
            writer.add_histogram("val_EFs", val_efs, epoch )

        print(f"[{epoch}] val_loss: {ave_val_loss:.6f}," +
                " accuracys: "+ " ".join(f"{x:.3f}" for x in val_accuracys ) + 
                ", precision: "+ " ".join(f"{x:.3f}" for x in val_precisions) + 
                ", recall: " + " ".join(f"{x:.3f}" for x in val_recalls ) +
                f", roc_auc: {val_auc:.3f},"+
                " EFs: " + " ".join(f"{x:.3f}" for x in val_efs ))

        if np.mean(loss_test_epoch) >= min_test_loss:
            n_unchanged_test_loss += 1
            if n_unchanged_test_loss>=patience:
                torch.save(model_dict_best, modelfn_best)
                print(f"Early stopping. Best epoch: {best_epoch}")
                break
        else:
            min_test_loss = np.mean(loss_test_epoch)
            n_unchanged_test_loss = 0
            best_epoch = epoch
            model_dict_best = {"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss, 
                    "test_loss": test_loss, 
                    "test_accuracys": test_accuracys,
                    "test_precisions": test_precisions,
                    "test_recalls": test_recalls,
                    "test_efs": test_efs,
                    "test_roc_auc": test_auc,
                    "val_accuracys": val_accuracys,
                    "val_precisions": val_precisions,
                    "val_recalls": val_recalls,
                    "val_efs": val_efs,
                    "val_roc_auc": val_auc,
                    "hit_ratio": args.hit_ratio
                    }
    writer.flush()
    writer.close()
    print("Training Done.")
    
    return modelfn_best


def get_args(i_iter: int, ratio: float, score_cutoff: float, config: dict):
    train_data_path = config['train_data_path']
    test_datafn = config["test_datafn"]
    val_datafn = config['val_datafn']
    train_datafn = []
    project_name = config['project_name']
    for i in range(1, i_iter+1):
        i_trainfn = os.path.join(train_data_path, f"{project_name}_train{i}_vs_results.aug.feather")
        if not os.path.exists(i_trainfn):
            i_trainfn = os.path.join(train_data_path, f"{project_name}_train{i}_vs_results.feather")
            if not os.path.exists(i_trainfn):
                raise Exception(f"Cannot find {i_trainfn}")
        train_datafn.append(i_trainfn)

    score_column = config["score_column"]
    prefix = config["prefix"]

    args = TrainArgs()
    
    model_path = os.path.join(config["model_path"], f"model_{i_iter}")
    prediction_path = os.path.join( config["prediction_path"], f"model_{i_iter}")
    new_smiles_dir = os.path.join(config["project_path"], "docking_smiles")
    new_traindata_dir = os.path.join(os.getcwd(), "dataset", f"vanilla_{prefix}_{score_column}", 
                                    "train_dataset")
        
    args_dict = {
        "train_datafn": train_datafn,
        "test_datafn": test_datafn,
        "validate_datafn": val_datafn,
        "dataset_type": "binaryclass", 
        "epochs": config['n_epochs'],
        "bias": config['use_bias'],
        "batch_size": 5000, 
        "patience": 3,
        "hit_ratio": ratio, 
        "score_cutoff": score_cutoff,
        "score_column": score_column,
        "prefix": prefix,
        "pytorch_seed": 16666,
        "dropout": 0.5,
        "task_names": ["hit_classification"],
        "metric": 'binary_cross_entropy',
        "extra_metrics": ['auc', 'accuracy', 'precisions', 'recalls', "enrichment_factor"],
        "log_frequency": 100,
        "weight_class": True,
        "model_path": model_path,
        "prediction_path": prediction_path,
        "new_smiles_path": new_smiles_dir,
        "new_traindata_dir": new_traindata_dir,
        "disable_progress_bar": not EnableBar
    }

    args.from_dict(args_dict)
    print("TrainArgs:", args)

    model_args = VanillaModelArgs()
    model_args_dict = {
        "nnodes": 3000,
        "nBits": 1024,
        "dataset_type": "binaryclass",
        "dropout":  0.5
    }
    model_args.from_dict(model_args_dict)
    print("Vanilla Model Args:", model_args)

    return args, model_args


def run_training(args, model_args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    writer = SummaryWriter(args.model_path)
    best_modelfn = train_vanilla(args, model_args, writer)
    return best_modelfn


def main(i_iter: int, configfn: dict):
    config = load_configfn(configfn)
    max_iter = config['max_iter']
    if i_iter > max_iter:
        print(f"max iteration {max_iter} has reached, stop.")
        return

    score_column = config['score_column']
    val_datafn = config['val_datafn']
    min_ratio = -4 #logspace 0.0001
    max_ratio = -1 #logspace 0.1
    val_df = pd.read_feather(val_datafn)
    scores_sorted = np.sort(val_df[score_column])
    ndata = len(scores_sorted)
    
    ratios = np.logspace(max_ratio , min_ratio , max_iter)
    print(f"Ratios used in determining the score cutoff: {ratios}")
    ratio = ratios[i_iter-1]
    cut_ndx = int(ratio*ndata)
    print(f"Using validation top {ratio:.4f} as hits, N of hits: {cut_ndx}, score column: {score_column}, score_cutoff: {scores_sorted[cut_ndx]:.2f}" )
    score_cutoff = scores_sorted[cut_ndx]
    args, model_args = get_args(i_iter, ratio, score_cutoff, config)
    
    best_modelfn = run_training(args, model_args)
    print("Current best model: ", best_modelfn)

def load_configfn(configfn):
    with open(configfn, 'rb') as f:
        config = orjson.loads(f.read())
    return config

if __name__ == "__main__":

    configfn = os.path.join("../",  "config_real_db.json")
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])

    main(i_iter, configfn)


