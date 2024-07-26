indir=../top_cluster_pdbs/iter8/top1000_pdbs_all
outdir=../top_cluster_pdbs/iter8/top1000_pdbs_all_ligand
mkdir -p $outdir 

find $indir/*.pdb -type f -print0 |
while read -r -d '' line
do
	molid=$(echo $line | cut -d '/' -f 5 | cut -d '.' -f 1)
	grep "LG1" $line > $outdir/$molid.lig.pdb
done 
