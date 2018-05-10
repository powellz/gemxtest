#!/usr/bin/env bash
pushd ./fcn
ln -s ../mergedinfo.yml .
DSA_PLATFORM=xilinx_kcu1500_dynamic_5_0  DSA=xilinx:kcu1500:dynamic:5_0  ../board_lsf.sh
popd
