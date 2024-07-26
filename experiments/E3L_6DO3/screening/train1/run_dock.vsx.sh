#!/bin/bash

outpath=$1
prefix=$2
flags_params=$3
ligand_list=$4


workdir=$PWD
RosettaCMD=$ROSETTAHOME/source/bin/rosetta_scripts.linuxgccrelease

$RosettaCMD \
@ $flags_params \
-s ../target/6DO3_0001.lig.pdb \
-extra_res_fa ../target/KLHDC2_c8_lig.params \
-gen_potential \
-overwrite \
-beta_cart \
-parser:protocol $workdir/dock_vsx.xml \
-parser:script_vars liglist=$ligand_list \
-no_autogen_cart_improper \
-multi_cool_annealer 10 \
-missing_density_to_jump \
-score::hb_don_strength hbdon_GENERIC_SC:1.45 \
-score::hb_acc_strength hbacc_GENERIC_SP2SC:1.19 \
-score::hb_acc_strength hbacc_GENERIC_SP3SC:1.19 \
-score::hb_acc_strength hbacc_GENERIC_RINGSC:1.19 \
-out:prefix $prefix \
-out:file:silent $outpath/"$prefix"out \
-out:file:scorefile $outpath/"$prefix"score.sc \
-mute all \
>& /dev/null

