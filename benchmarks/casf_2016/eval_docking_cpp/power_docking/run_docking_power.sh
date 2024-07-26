database=./CASF-2016/decoys_docking/
extra="eval_docking_cpp_relax_lig_mcentropy_ligcst1"
python docking_power.py -c CoreSet.dat -s ../gathered_results/${extra}_score -r $database -p 'negative' -l 2 -o scores_$extra > scores_docking_power_$extra.out
