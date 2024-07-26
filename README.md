# OpenVS

Codes for "An artificial intelligence accelerated virtual screening platform for drug discovery". [https://www.biorxiv.org/content/10.1101/2024.03.28.587262v1]

## Installation
Software installation time is less than 2 hours on a regular desktop

### Install Rosetta
Please follow the instructions in https://github.com/RosettaCommons/rosetta
After the installation, run `export ROSETTAHOME=your_local_path`, remember to replace `your_local_path` with your the actual installation path.

### Install Dimorphite-DL 1.2.4
Please follow the instructions in https://github.com/durrantlab/dimorphite_dl
After the installation, run `export DIMORPHITE=your_local_path`, remember to replace `your_local_path` with your the actual installation path.

### Install OpenVS
conda create -n openvs python=3.9
conda install --file requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -e .

To install CSD Python API, follow https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html


## Small molecule library
The drug-like centroid library can be downloaded at: https://files.ipd.uw.edu/pub/OpenVS/centroids.tgz

ZINC22 3d can be downloaded from: https://cartblanche22.docking.org/

Enamine REAL library can be downloaded from: https://enamine.net/compound-collections/real-compounds/real-database

Once downloaded, put the library in `./databases`

## Usage

`openvs` folder contains utility modules for OpenVS platform 

`benchmarks` folder contains scripts to run benchmarks on two datasets:
```
    "benchmarks/casf_2016" CASF2016 benchmark
        "benchmarks/casf_2016/eval_docking_cpp" Docking power. Please see README file in the folder for instructions.
        "benchmarks/casf_2016/eval_screening_cpp" Sreening power. Please see README file in the folder for instructions.
        "benchmarks/casf_2016/holo_relax_apo_cstw1" Folder contains constraint relaxed holo pdb file from CASF2016.
        "benchmarks/casf_2016/params_new" Folder contains the params file for the ligands in CASF2016.
    "benchmarks/dud" DUD benchmark
        Please see README file in the folder for instructions.
```

`experiments` folder contains two virtual screning experiments performed in this work
    Please see the README file in the folder for instructions.

`databases` folder contains example small molecule database files

`scripts` folder contains scripts to preprocess and generate fingerprints for the small molecules in the database. 