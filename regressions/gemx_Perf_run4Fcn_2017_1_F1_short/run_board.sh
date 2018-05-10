#!/usr/bin/env bash
pushd ./fcn
ln -s ../mergedinfo.yml .
DSA_PLATFORM=xilinx_aws-vu9p-f1_4ddr-xpr-2pr_4_0  DSA=xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4_0  ../board_lsf.sh
popd
