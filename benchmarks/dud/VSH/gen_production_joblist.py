from __future__ import print_function

import os,sys
import glob
import subprocess as sp

def write_sbatchfn(sbatchfn, submitfn, joblistfn, n_jobs, job_name="arrayjobs", queue='cpu'):

    if queue == 'cpu':
        sbatch_content = """#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=20g
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --time 5:30:00
#SBATCH --job-name={jobname}
#SBATCH -o output.{jobname}.log

source ~/.bashrc
CMD=$(head -$SLURM_ARRAY_TASK_ID {joblistfn} | tail -1)
exec ${{CMD}}
""".format(jobname=job_name, joblistfn=joblistfn)
        with open(sbatchfn, 'w') as outf:
            outf.write(sbatch_content)
        print("Wrote: %s"%sbatchfn)
        
        submit_content = """#!/bin/bash
sbatch -a 1-{} {}
""".format( n_jobs,os.path.basename(sbatchfn) )
        with open(submitfn, 'w') as outf:
            outf.write(submit_content)
        print("Wrote: %s"%submitfn)


def gen_cmdlist_dud_target(trg, indir, outrootdir, batchsize=50, overwrite = False):
    in_dir = os.path.join(indir, f"{trg}_tar_inputs_chunk{batchsize}")
    pattern = os.path.join(in_dir, "*.tar")
    tarfn_list = sorted(glob.glob(pattern))

    cmd_lines = []
    subdir = f"results_vsh"
    app = "./run_dock.sh"
    n_runs = 3

    for i in range(n_runs):
        runid = "%d."%i
        outpath = os.path.join(outrootdir, subdir, f"{trg}_results", f"run{i}")
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for tarfn in tarfn_list:
            tarfn_name = os.path.basename(tarfn)
            fields = tarfn_name.split(".")[0].split('_')
            ndx1 = fields[-1]
            outfn = os.path.join( outpath, f"{trg}_{ndx1}.{i}.out" )
            if os.path.exists(outfn) and not overwrite:
                print("%s exists, skip..."%outfn)
                continue
            ligand_list_name = f"ligands_list_{trg}_{ndx1}.txt"
            ligand_listfn = os.path.join(in_dir, ligand_list_name)
            prefix = f"{trg}_{ndx1}.{runid}"
            if not os.path.exists(ligand_listfn):
                print("Cannot find {}, skip..".format(ligand_listfn))
                continue
            cmd = f"{app} {outpath} {prefix} {tarfn} {ligand_listfn} {trg}\n"
            #print(cmd.strip())
            cmd_lines.append(cmd)
    return cmd_lines


def gen_cmdlistfn_for_multiprocess(cmdlist, outdir_cmdlistfn, cmd_size = 5*16):

    if not os.path.exists(outdir_cmdlistfn):
        os.makedirs(outdir_cmdlistfn)
        print("Made directory: %s"%outdir_cmdlistfn)
    else:
        pattern = os.path.join(outdir_cmdlistfn, "*.cmdlist")
        cmd = "rm %s"%pattern
        print(cmd)
        p = sp.Popen(cmd, shell=True)
        p.communicate()

    counter = 0
    contents = []
    cmdlistfns = []
    
    for cmd in cmdlist:
        contents.append(cmd)
        if len(contents) == cmd_size:
            outfn_ndx = counter//cmd_size
            cmdlistfn = f"dud_{outfn_ndx}.cmdlist"
            outfn_cmdlist = os.path.join(outdir_cmdlistfn, cmdlistfn)
            with open(outfn_cmdlist, "w") as outf:
                outf.writelines(contents)
            contents = []
            print("Wrote: %s"%outfn_cmdlist)
            cmdlistfns.append(outfn_cmdlist)
        counter += 1

    if len(contents) != 0:
        outfn_ndx = counter//cmd_size
        cmdlistfn = f"dud_{outfn_ndx}.cmdlist"
        outfn_cmdlist = os.path.join(outdir_cmdlistfn, cmdlistfn)
        with open(outfn_cmdlist, "w") as outf:
            outf.writelines(contents)
        contents = []
        print("Wrote: %s"%outfn_cmdlist)
        cmdlistfns.append(outfn_cmdlist)
    return cmdlistfns

def gen_joblist_production(cmdlistfns, queue='ckpt', extra=""):

    outdir = "./"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print("Make new directory", outdir)
    joblist_fn = os.path.join(outdir, "dud_dock.joblist" )

    joblist_lines = []
    exec_scripts = os.path.join('parallel :::: ')

    for cmdlistfn in cmdlistfns:
        cmd = "%s %s\n"%(exec_scripts, cmdlistfn)
        print(cmd.strip())
        joblist_lines.append(cmd)

    with open(joblist_fn, 'w') as outf:
        outf.writelines(joblist_lines)
    print("Wrote: %s"%joblist_fn)
    n_jobs = len(joblist_lines)

    sbatchfn = os.path.join( outdir, "sbatch_dock_arrayjobs.sh" )
    submitfn = os.path.join( outdir, "submit_arrayjobs.sh" )
    jobname = f"vsh_dud{extra}"
    joblistfn = os.path.basename(joblist_fn)
    write_sbatchfn(sbatchfn, submitfn, joblistfn, n_jobs, jobname, queue)


def gen_joblist_dud():
    trgfn = "../lists/trglist.txt"
    with open(trgfn, 'r') as infh:
        trglist = [l.strip() for l in infh]
    trglist = set(trglist) 

    overwrite = False
    indir = os.path.join(os.getcwd(), "inputs")
    outrootdir = os.path.join(os.getcwd(), "outputs")
    batchsize=10
    cmdlist = []
    for trg in trglist:
        cmdlist.extend( gen_cmdlist_dud_target(trg, indir, outrootdir, batchsize, overwrite) )
    outdir_cmdlist = "./cmdlistfns"
    cmd_size = 1*5
    queue = 'cpu'
    cmdlistfns = gen_cmdlistfn_for_multiprocess(cmdlist, outdir_cmdlist, cmd_size )
    extra=""
    gen_joblist_production(cmdlistfns, queue, extra)
    

if __name__ == '__main__':
    gen_joblist_dud()    

