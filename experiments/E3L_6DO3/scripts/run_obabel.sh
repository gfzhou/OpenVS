smifn=$1
tempprefix=$2
mol2fn=$3
infer_H=$4

if [[ "$infer_H" -eq 1 ]]
then
  echo $infer_H Infer protonation using obabel.
  obabel $smifn -O $tempprefix.1.mol2 -e -d --gen3d --conformer -nconf 100 --score energy --partialcharge mmff94 -xl >& /dev/null
  obabel $tempprefix.1.mol2 -O $mol2fn -e -p 7.0 --partialcharge mmff94 -xl --minimize --steps 2000 --sd >& /dev/null
  rm $tempprefix.1.mol2
else
  echo $infer_H Input smiles are already protonated.
  obabel $smifn -O $tempprefix.1.pdb -e --gen3d --conformer -nconf 100 --score energy --partialcharge mmff94 -xl >& /dev/null
  obabel $tempprefix.1.pdb -O $mol2fn -e -h --minimize --steps 2000 --sd --partialcharge mmff94 --title "" --append "COMPND" -xl >& /dev/null 
  rm $tempprefix.1.pdb
fi

