#!/usr/bin/env bash
pushd ./pca
ln -s ../mergedinfo.yml .
DSA_PLATFORM=xilinx_vcu1525_dynamic_5_0  DSA=xilinx:vcu1525:dynamic:5_0  ../board_lsf.sh
popd