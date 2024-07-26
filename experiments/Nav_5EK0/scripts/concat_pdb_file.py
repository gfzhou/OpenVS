import os,sys
from glob import glob
import subprocess as sp

def parse_pdbline(line):
    l = line.strip()
    rcdtype = l[:6].strip()
    atomid = int(l[6:11].strip())
    atomname = l[12:16].strip()
    res3l = l[17:20].strip()
    chainid = l[21]
    seqid = l[22:26].strip()
    if seqid:
        seqid = int(l[22:26].strip())
    else:
        seqid = 1
    x = float(l[30:38].strip())
    y = float(l[38:46].strip())
    z = float(l[46:54].strip())
    occ = float(l[54:60].strip())
    bfac = float(l[60:66].strip())
    element = l[76:78].strip()
    charge = l[78:80]
    return rcdtype, atomid, atomname, res3l, chainid, seqid, x, y, z, occ, bfac, element, charge

class Atom(object):
    def __init__(self, rcdtype, atomid, atomname, res3l, chainid, seqid, x,y,z, occ, bfac, element, charge):
        self.rcdtype = rcdtype
        self.atomid = int(atomid)
        self.name = atomname
        self.res3l = res3l
        self.chainid = chainid
        self.seqid = int(seqid)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.occ = float(occ)
        self.bfac = float(bfac)
        self.element = element
        self.charge = charge
    def set_coords(self, coords):
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]
    
    def ister(self):
        return self.rcdtype.strip().lower() == 'ter'


class Molecule(object):
    def __init__(self):
        self.atoms = []

    def n_atoms(self):
        return len(self.atoms)

    def FromPDBFile(self, pdbfn):
        with open(pdbfn, 'r') as infh:
            for l in infh:
                if l.startswith('ATOM') or l.startswith('HETATM'):
                    rcdtype, atomid, atomname, res3l, chainid, seqid, x, y, z, occ, bfac, element, charge = parse_pdbline(l)
                    atom = Atom( rcdtype, atomid, atomname, res3l, chainid, seqid, x, y, z, occ, bfac, element, charge )
                    self.atoms.append(atom)
    
    def WritePDBFile(self, pdbfn):
        if len(self.atoms) == 0:
            return False
        content = []
        for atm in self.atoms:
            if atm.ister():
                line = "TER\n"
                #line = f"{'TER':6}{atm.atomid:>5d} {'':<4} {atm.res3l:>3}  {atm.chainid}{atm.seqid:>4d}\n"
            else:
                line = f"{atm.rcdtype:6}{atm.atomid:>5d} {atm.name:<4} {atm.res3l:>3} {atm.chainid}{atm.seqid:>4d}    {atm.x:>8.3f}{atm.y:>8.3f}{atm.z:>8.3f}{atm.occ:>6.2f}{atm.bfac:>6.2f}       {'   '}{atm.element:>2}{atm.charge}\n"
            content.append(line)
        with open(pdbfn, 'w') as outf:
            outf.writelines(content)
        return True

def concat(pdbfn1, pdbfn2, outfn):
    m1 = Molecule()
    m1.FromPDBFile(pdbfn1)
    m2 = Molecule()
    m2.FromPDBFile(pdbfn2)
    m1_atmid = m1.atoms[-1].atomid
    m1_seqid = m1.atoms[-1].seqid
    m2_seqid1 = m2.atoms[0].seqid
    c = 1
    for atm in m2.atoms:
        atm.atomid = m1_atmid + c 
        c += 1
        atm.seqid += m1_seqid - m2_seqid1 + 1
        atm.occ = 1.0
        atm.chainid = 'X'
        m1.atoms.append(atm)
    m1.WritePDBFile(outfn)
    print(f"Wrote: {outfn}")

def count_atomline(infn):
    n=0
    with open(infn, 'r') as infh:
        for l in infh:
            if l.startswith('ATOM'): n += 1
    return n

def concat_pdbs_folder(indir, outdir):
    os.makedirs(outdir, exist_ok=True)
    patt = os.path.join(indir, "*.pdb")
    pdbfns = glob(patt)
    recpdbfn = "../target/VSD-5EK0.lig_0001.pdb"
    for ligpdbfn in pdbfns:
        fname = os.path.basename(ligpdbfn)
        outpdbfn = os.path.join(outdir, fname)
        if count_atomline(ligpdbfn) > 50:
            cmd = f"cp {ligpdbfn} {outpdbfn}"
            p = sp.Popen(cmd, shell=True)
            p.communicate()
        else:
            concat(recpdbfn, ligpdbfn, outpdbfn)

        
if __name__ == "__main__":
    i_iter=7
    indir = f"../top_cluster_pdbs/iter{i_iter}/top1000_pdbs_all"
    outdir = f"../top_cluster_pdbs/iter{i_iter}/top1000_pdbs_all_holo"
    concat_pdbs_folder(indir,outdir)
  
