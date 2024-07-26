import os,sys
import tarfile


def gen_flags_target_tar(prefix, src_tar, outdir, batch_size=100):
    
    base_tarfn = os.path.basename(src_tar).split('.')[0]
    #print("base tar filename:", base_tarfn)
    base_ndx = int(base_tarfn.split('_')[-1])
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
            tarfilepath = os.path.join(outdir, "tar_params_%s_%d_%d.tar"%(prefix, base_ndx, i))
            outligandfn = os.path.join(outdir, "ligands_list_%s_%d_%d.txt"%(prefix, base_ndx, i))
            outparamfn = os.path.join(outdir, "flags_params_%s_%d_%d.flags"%(prefix, base_ndx, i))
            ligandlines = []
            flagslines = []

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
                    #print(tarfilepath, member.name, line)
                    ligandID = line.decode().split()[1]
                    
                    ligandlines.append(ligandID+"\n")
            print("Saved: %s"%tarfilepath)
            flagslines.append("-in:file:extra_res_fa "+tarfilepath+"\n")
            with open(outligandfn, 'w') as outf:
                outf.writelines(ligandlines)
            print("Saved: %s"%outligandfn)
            with open(outparamfn, 'w') as outf:
                outf.writelines(flagslines)
            print("Saved: %s"%outparamfn)

def gen_flags_deepdock(batch_size=100):
    prefix="validation" 
    project_name="E3L_6DO3"
    src_tar = os.path.join("params", f"{prefix}_params_0.tar")
    outdirname = f"{project_name}_{prefix}_chunk{batch_size}"
    outpath = os.path.join(os.getcwd(), "inputs", f"{prefix}set", outdirname)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    gen_flags_target_tar(prefix, src_tar, outpath, batch_size)
    
if __name__=="__main__":
    gen_flags_deepdock(2)

