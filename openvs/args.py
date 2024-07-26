import os,sys
from tap import Tap
from typing import Any, Callable, List, Tuple, Union
from typing_extensions import Literal

class ExtractSmilesArgs(Tap):
    n_iter: int
    n_data: int
    outdir: str
    project_name: str
    trainfn: str
    testfn: str
    validatefn: str
    datarootdir: str

class VanillaModelArgs(Tap):
    nnodes: int = 3000
    nBits: int = 1024
    dataset_type: Literal["binaryclass", "multiclass", "regression"]
    dropout: float = 0.5
    nlayers: int = 2

class TrainArgs(Tap):
    modelid: str = "0"
    i_iter: int = 1
    train_datafn: str = None
    test_datafn: str = None
    validate_datafn: str = None
    hit_ratio: float = 0.0
    score_cutoff: float = 0.0
    prefix: str = ""
    device: str = "cuda"
    batch_size: int = 50
    epochs: int = 10
    rand_seed: int = 66666
    log_frequency: int = 500
    weight_class: bool = False
    class_weights: List=[1,1,1,1]
    patience: int = 5
    disable_progress_bar : bool = False
    inferenceDropout: bool = False
    

class EvalArgs(Tap):
    topNs: List = [10, 100, 1000, 10000]
    thresholds: List = [0.2, 0.35, 0.5]
    target_threshold: float = None
    target_recall: float = 0.9 #only used in validation set evaluation
    rand_active_prob: float
    dataset_type: Literal["test", "validate"]
    disable_progress_bar : bool = False

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

