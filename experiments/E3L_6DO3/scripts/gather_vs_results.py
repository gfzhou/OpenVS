import os,sys
import tarfile
import subprocess as sp
import pandas as pd
from openvs.utils.utils import load_configfn
from openvs.utils.rosetta_utils import read_log_fstream
    

def gather_results_tarhelper(intarfn, tags, dtypes, outfn, 
                                datafile='silent', column_name=None, 
                                remove_duplicates=True):
    if datafile == "silent":
        extension = ".out"
    print(intarfn)
    all_values = []
    with tarfile.open(intarfn, 'r') as tarfh:
        for member in tarfh.getmembers():
            if not member.name.endswith(extension):
                continue
            fh=tarfh.extractfile(member)
            valid_tags, values = read_log_fstream(fh, tags, dtypes)
            if len(tags) != len(valid_tags):
                raise(f"Valid tags: {valid_tags} doesn't match specified tags {tags}!")
            all_values.extend(values)
    df = pd.DataFrame(all_values, columns=valid_tags)
    if remove_duplicates and column_name is not None:
        df.drop_duplicates(subset=column_name, keep="first", inplace=True, ignore_index=True)
    if outfn.endswith(".csv"):
        df.to_csv(outfn, index=False)
    elif outfn.endswith(".feather"):
        df.to_feather(outfn)
    else:
        raise(f"{outfn} foramt is not supported.")
    print(f"Number non duplicated data: {len(df)}")
    print("Saved: %s"%outfn)

def gather_results(prefix, datadir, tempdir="/net/scratch/temp"):
    # we will use feather files
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    tarfname = f"{prefix}.tar"
    tarfn = os.path.join(datadir, tarfname)
    tags = ["score", "complexscore", "dG", "ligandname", "description"]
    dtypes = [float, float, float, str, str]

    outfn = os.path.join(datadir, f"{prefix}_vs_results.feather")

    if os.path.exists(outfn):
        print("%s exists!"%outfn)
        return
    if os.path.exists(tarfn):
        print(f"Processing {tarfn}")
        gather_results_tarhelper(tarfn, tags, dtypes, outfn, 
                                    column_name="ligandname", 
                                    remove_duplicates=True)
        return
    #prefix is also the subdir name
    tardir = os.path.join(datadir, prefix)
    if not os.path.exists(tardir):
        print("Cannot find tar dir %s"%tardir)
        return
    tarsplitfns = os.path.join(tardir, f"{tarfname}*") 
    temp_tarfn = os.path.join(tempdir, tarfname)
    if not os.path.exists(temp_tarfn):
        cmd = "cat %s > %s"%(tarsplitfns, temp_tarfn)
        print(cmd)
        p = sp.Popen(cmd, shell=True)
        p.communicate()
    gather_results_tarhelper(temp_tarfn, tags, dtypes, outfn, column_name="ligandname", 
                                    remove_duplicates=True)

def gather_substructure():
    configfn = os.path.join("../", "config_zinc22_db.json")
    config = load_configfn(configfn)
    project = config['project_name']
    datadir = config['result_path']
    tempdir = config['concat_tar_path']
    dirnames = [f"{project}_substructure"]
    print(dirnames)
    for prefix in dirnames:
        gather_results(prefix, datadir, tempdir=tempdir)

def main():
    if len(sys.argv) <2:
        print("Usage: python this.py iter")
        raise
    else:
        i_iter = int(sys.argv[1])
    if i_iter == 1:
        configfn = os.path.join("../", "config_clusterdb.json")
    else:
        configfn = os.path.join("../", "config_real_db.json")
    config = load_configfn(configfn)
    project = config['project_name']
    datadir = config['result_path']
    tempdir = config['concat_tar_path']
    if i_iter == 1:
        dirnames = [f"{project}_test", f"{project}_validation", f"{project}_train1"]
    else:
        dirnames = [f"{project}_train{i_iter}"]
    print(dirnames)
    for prefix in dirnames:
        gather_results(prefix, datadir, tempdir=tempdir)

if __name__ == "__main__":
    main()
    #gather_substructure()

