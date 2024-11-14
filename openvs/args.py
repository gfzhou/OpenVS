'''Typed arguments defination for argparse type checking and code completion.'''

import os,sys
from tap import Tap
from typing import Any, Callable, List, Optional, Tuple, Union
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
    '''Typed args for Vanilla model.'''
    nnodes: int = 3000
    '''Neuron nodes number in one layer'''
    nBits: int = 1024
    '''Length of morgan fingerprint vector.'''
    dataset_type: Literal["binaryclass", "multiclass", "regression"]
    '''Predict form.'''
    dropout: float = 0.5
    '''Dropout factor in dropout layer.'''
    nlayers: int = 2
    '''Number of same layers.'''


class TrainArgs(Tap):
    '''Typed args for training mode.'''
    modelid: str = "0"
    i_iter: int = 1
    train_datafn: Optional[str] = None
    test_datafn: Optional[str] = None
    validate_datafn: Optional[str] = None
    hit_ratio: float = 0.0
    score_cutoff: float = 0.0
    prefix: str = ""
    device: str = "cuda"
    batch_size: int = 50
    epochs: int = 10
    rand_seed: int = 66666
    log_frequency: int = 500
    weight_class: bool = False
    class_weights: List[float] = [1, 1, 1, 1]
    patience: int = 5
    disable_progress_bar: bool = False
    inferenceDropout: bool = False


class EvalArgs(Tap):
    topNs: List[int] = [10, 100, 1000, 10000]
    thresholds: List[float] = [0.2, 0.35, 0.5]
    target_threshold: Optional[float] = None
    target_recall: float = 0.9 #only used in validation set evaluation
    rand_active_prob: float
    dataset_type: Literal["test", "validate"]
    disable_progress_bar : bool = False


class PredictArgs(Tap):
    '''Typed args for predicting mode.'''
    modelfn: Optional[str] = None
    database_type: Optional[str] = None
    database_path: Optional[str] = None
    prediction_path: Optional[str] = None
    disable_progress_bar: bool = True
    '''Whether to disable progresss bar.'''
    batch_size: int = 10000
    outfileformat: str = "feather"
    '''Extension name of the output file.'''
    run_platform: str="auto" #Literal["gpu", "slurm", "auto"], I need "auto" to be default
    i_iter: int
