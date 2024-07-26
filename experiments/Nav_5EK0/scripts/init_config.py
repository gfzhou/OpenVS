import os
import orjson
import subprocess as sp


#project global settings:
ROOT_PATH = sp.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
EXP_PATH = os.path.join(ROOT_PATH, "experiments")
PROJ_NAME = "Nav_5EK0"
SCRATCH_PATH = os.path.join(EXP_PATH, PROJ_NAME, "scratch")
RESULTS_PATH = os.path.join(EXP_PATH, PROJ_NAME, "screening", "outputs")
DB_PATH = os.path.join(ROOT_PATH, "databases")



def init_prod_configfn():
    pass


def init_configfn_cluster_database():
    ## specific to cluster db
    params_basedir = os.path.join(DB_PATH, "centroids")
    index_dir_params = os.path.join(DB_PATH, "centroids", "index")
    smiles_path = os.path.join(DB_PATH, "centroids", "smiles")
    fps_path = os.path.join(DB_PATH, "centroids", "fingerprints")

    project_path = os.path.join(EXP_PATH, PROJ_NAME)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    config_path = os.path.join(project_path, "config_clusterdb.json")
    prediction_path = os.path.join( SCRATCH_PATH, "predictions")

    ntrain = 20
    ntest = 20
    nval = 20
    max_iter = 10

    prefix = PROJ_NAME
    results_path = RESULTS_PATH
    train_data_path = results_path
    test_datafn = os.path.join(
        results_path, f"{prefix}_test_vs_results.aug.feather")
    val_datafn = os.path.join(
        results_path, f"{prefix}_validation_vs_results.aug.feather")
    model_path = os.path.join(project_path, "models")
    new_smiles_path = project_path
    new_traindata_path = project_path

    concat_tar_path = os.path.join(SCRATCH_PATH, "results")
    mol2_path = os.path.join(SCRATCH_PATH, "mol2s")

    json_dict={
                "project_name": PROJ_NAME,
                "project_path": project_path,
                "params_basedir": params_basedir,
                "index_dir_params": index_dir_params,
                "smiles_path": smiles_path,
                "fps_path": fps_path,
                "database_type": "full",
                "train_size": ntrain,
                "test_size": ntest,
                "val_size": nval,
                "max_iter": max_iter,
                "train_data_path": train_data_path,
                "test_datafn": test_datafn,
                "val_datafn": val_datafn,
                "dataset_type": "binaryclass",
                "n_epochs": 30,
                "use_bias": True,
                "batch_size": 10000,
                "patience": 3,
                "score_column": 'dG',
                "prefix": prefix,
                "pytorch_seed": 16666,
                "dropout": 0.5,
                "task_names": ["hit_classification"],
                "metric": 'binary_cross_entropy',
                "extra_metrics": ['auc', 'accuracy', 'precisions', 'recalls', "enrichment_factor"],
                "log_frequency": 100,
                "weight_class": True,
                "result_path": results_path,
                "model_path": model_path,
                "prediction_path": prediction_path,
                "new_smiles_path": new_smiles_path,
                "new_traindata_path": new_traindata_path,
                "disable_progress_bar": True,
                "extract_rawmol2_path": mol2_path,
                "project_tempdir": SCRATCH_PATH,
                "concat_tar_path": concat_tar_path
                }

    with open(config_path, "wb") as outfh:
        outfh.write(orjson.dumps(json_dict, option=orjson.OPT_INDENT_2))
    print(f"Saved: {config_path}")


def init_configfn_real_database():
    
    project_path = os.path.join(EXP_PATH, PROJ_NAME)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    config_path = os.path.join(project_path, "config_real_db.json")
    smiles_path = os.path.join(DB_PATH, "real", "smiles", "split")
    fps_path = os.path.join(DB_PATH, "real", "fingerprints", "split")
    prediction_path = os.path.join( SCRATCH_PATH, "predictions_real_db")

    ntrain = 20
    ntest = 20
    nval = 20
    max_iter = 10

    prefix = PROJ_NAME

    results_path = RESULTS_PATH
    train_data_path = results_path
    test_datafn = os.path.join(
        results_path, f"{prefix}_test_vs_results.aug.feather")
    val_datafn = os.path.join(
        results_path, f"{prefix}_validation_vs_results.aug.feather")
    model_path = os.path.join(project_path, "models")
    new_smiles_path = project_path
    new_traindata_path = project_path

    concat_tar_path = os.path.join(SCRATCH_PATH, "results")
    mol2_path = os.path.join(SCRATCH_PATH, "mol2s")


    json_dict={
                "project_name": PROJ_NAME,
                "project_path": project_path,
                "smiles_path": smiles_path,
                "fps_path": fps_path,
                "database_type": "real",
                "train_size": ntrain,
                "test_size": ntest,
                "val_size": nval,
                "max_iter": max_iter,
                "train_data_path": train_data_path,
                "test_datafn": test_datafn,
                "val_datafn": val_datafn,
                "dataset_type": "binaryclass",
                "n_epochs": 30,
                "use_bias": True,
                "batch_size": 10000,
                "patience": 3,
                "score_column": 'dG',
                "prefix": prefix,
                "pytorch_seed": 16666,
                "dropout": 0.5,
                "task_names": ["hit_classification"],
                "metric": 'binary_cross_entropy',
                "extra_metrics": ['auc', 'accuracy', 'precisions', 'recalls', "enrichment_factor"],
                "log_frequency": 100,
                "weight_class": True,
                "result_path": results_path,
                "model_path": model_path,
                "prediction_path": prediction_path,
                "new_smiles_path": new_smiles_path,
                "new_traindata_path": new_traindata_path,
                "disable_progress_bar": True,
                "mol2_path": mol2_path,
                "project_tempdir": SCRATCH_PATH,
                "concat_tar_path": concat_tar_path
                }

    with open(config_path, "wb") as outfh:
        outfh.write(orjson.dumps(json_dict, option=orjson.OPT_INDENT_2))
    print(f"Saved: {config_path}")

def init_configfn_zinc22_database():
    
    project_path = os.path.join(EXP_PATH, PROJ_NAME)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    config_path = os.path.join(project_path, "config_zinc22_db.json")
    smiles_path = "" # not using smiles of zinc22
    zinc22_mol2_path = os.path.join(DB_PATH, "zinc", "zinc22", "zinc22") # use the raw mols from zinc22, tgz format
    fps_path = os.path.join(DB_PATH, "zinc", "zinc22", "fingerprints")
    prediction_path = os.path.join( SCRATCH_PATH, "predictions_zinc22_db")

    ntrain = 20
    ntest = 20
    nval = 20
    max_iter = 10

    prefix = PROJ_NAME

    results_path = RESULTS_PATH
    train_data_path = results_path
    test_datafn = os.path.join(
        results_path, f"{prefix}_test_vs_results.aug.feather")
    val_datafn = os.path.join(
        results_path, f"{prefix}_validation_vs_results.aug.feather")
    model_path = os.path.join(project_path, "models")
    new_smiles_path = project_path
    new_traindata_path = project_path

    mol2_path = os.path.join(SCRATCH_PATH, "mol2s")


    json_dict={
                "project_name": PROJ_NAME,
                "project_path": project_path,
                "smiles_path": smiles_path,
                "zinc22_mol2_path": zinc22_mol2_path,
                "fps_path": fps_path,
                "database_type": "zinc22",
                "train_size": ntrain,
                "test_size": ntest,
                "val_size": nval,
                "max_iter": max_iter,
                "train_data_path": train_data_path,
                "test_datafn": test_datafn,
                "val_datafn": val_datafn,
                "dataset_type": "binaryclass",
                "n_epochs": 30,
                "use_bias": True,
                "batch_size": 10000,
                "patience": 3,
                "score_column": 'dG',
                "prefix": prefix,
                "pytorch_seed": 16666,
                "dropout": 0.5,
                "task_names": ["hit_classification"],
                "metric": 'binary_cross_entropy',
                "extra_metrics": ['auc', 'accuracy', 'precisions', 'recalls', "enrichment_factor"],
                "log_frequency": 100,
                "weight_class": True,
                "result_path": results_path,
                "model_path": model_path,
                "prediction_path": prediction_path,
                "new_smiles_path": new_smiles_path,
                "new_traindata_path": new_traindata_path,
                "disable_progress_bar": True,
                "mol2_path": mol2_path,
                "project_tempdir": SCRATCH_PATH,
                "concat_tar_path": RESULTS_PATH
                }

    with open(config_path, "wb") as outfh:
        outfh.write(orjson.dumps(json_dict, option=orjson.OPT_INDENT_2))
    print(f"Saved: {config_path}")

def main(database=""):

    if database == "cluster":
        init_configfn_cluster_database()
    elif database == "real":
        init_configfn_real_database()
    elif database == "zinc22":
        init_configfn_zinc22_database()

if __name__ == "__main__":
    database = "zinc22"
    main(database)
    database = "cluster"
    main(database)
