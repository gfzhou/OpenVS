#!/bin/bash

outpath=$1
prefix=$2
paramsfn=$3
ligand_list=$4
trg=$5

workdir=$PWD
ROSETTA=$ROSETTAHOME/main/source/bin/rosetta_scripts.linuxgccrelease

if [ ! -d $outpath ]; then
    mkdir $outpath
fi

if [ -f $outpath/"$prefix"out ]; then
	echo $outpath/"$prefix"out exists
	exit
fi


$ROSETTA \
-s ../receptors_dud/"$trg".holo.pdb \
-extra_res_fa ../receptors_dud/"$trg".lig.params \
-extra_res_fa $paramsfn \
-gen_potential \
-overwrite \
-beta_cart \
-no_autogen_cart_improper \
-missing_density_to_jump \
-multi_cool_annealer 10 \
-parser:protocol $workdir/dock.xml \
-score::hb_don_strength hbdon_GENERIC_SC:1.45 \
-score::hb_acc_strength hbacc_GENERIC_SP2SC:1.19 \
-score::hb_acc_strength hbacc_GENERIC_SP3SC:1.19 \
-score::hb_acc_strength hbacc_GENERIC_RINGSC:1.19 \
-parser:script_vars liglist=$ligand_list \
-out:prefix $prefix \
-out:file:silent $outpath/"$prefix"out \
-out:file:scorefile $outpath/"$prefix"score.sc \
-mute all \
>& /dev/null
