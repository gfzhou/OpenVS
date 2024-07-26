import os,sys
import tarfile
from pathlib import Path


def gen_flags_target_tar(trg, src_tar, outdir, batch_size=100, overwrite=False):
    members = []
    with tarfile.open(src_tar, 'r') as intar:
        raw_members = intar.getmembers()
        for member in raw_members:
            if not member.isfile():
                continue
            if ".params" not in member.name:
                continue
            if os.path.basename(member.name).startswith("._"):
                continue
            members.append(member)
        begins =  range(0,len(members),batch_size)
        for i, beg in enumerate(begins):
            sub_members=members[beg:beg+batch_size]
            tarfilepath = os.path.join(outdir, f"tar_params_{trg}_{i}.tar")
            outligandfn = os.path.join(outdir, f"ligands_list_{trg}_{i}.txt")
            ligandlines = []
            if not overwrite and os.path.exists(tarfilepath) and os.path.exists(outligandfn):
                print(f"{tarfilepath} exists.")
                print(f"{outligandfn} exists.")
                print("skip.")
                continue

            with tarfile.open(tarfilepath, "w") as outtar:
                for member in sub_members:
                    if not member.isfile():
                        continue
                    if not member.name.endswith(".params"):
                        continue
                    if os.path.basename(member.name).startswith("._"):
                        continue
                    f = intar.extractfile(member)
                    outtar.addfile(member, f)
                    f = intar.extractfile(member)
                    line = f.readlines()[0].strip()
                    ligandID = line.decode().split()[1]
                    
                    ligandlines.append(ligandID+"\n")
            print("Saved: %s"%tarfilepath)
            with open(outligandfn, 'w') as outf:
                outf.writelines(ligandlines)
            print("Saved: %s"%outligandfn)

def gen_flags_dud(batch_size=100):
    trgfn = "../lists/trglist.txt"
    with open(trgfn, 'r') as infh:
        trglist = [l.strip() for l in infh]
    tar_params_path = os.path.join(Path(__file__).parents[1], "params_raw")
    for trg in trglist:
        src_tarfn = os.path.join(tar_params_path, f"{trg}.tar")
        if not os.path.exists(src_tarfn):
            print(f"{src_tarfn} not found.")
            continue
        
        outdirname = f"{trg}_tar_inputs_chunk{batch_size}"
        outpath = os.path.join( os.getcwd(), "inputs", outdirname)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        gen_flags_target_tar(trg, src_tarfn, outpath, batch_size)

if __name__=="__main__":
    gen_flags_dud(10)

