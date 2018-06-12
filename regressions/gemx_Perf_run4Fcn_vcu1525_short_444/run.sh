#!/usr/bin/env bash

unset SDSOC_VIVADO
unset SDSOC_VIVADO_HLS
unset SDX_VIVADO
unset SDX_VIVADO_HLS
unset XILINX_VIVADO_HLS
unset SDSOC_SDK
unset XILINX_SDK
unset PLATFORM_REPO_PATHS

export XILINX_SDX=/proj/xbuilds/2017.4_released/installs/lin64/SDx/2017.4
if [ ! -e $XILINX_SDX ]; then
  export XILINX_SDX=/proj/rdi-xco/xbuilds/2017.4_released/installs/lin64/SDx/2017.4
fi
source $XILINX_SDX/settings64.sh
env
which xocc vivado_hls vivado

pushd ./fcn 
make out_host/gemx_api_fcn_multiInstr.exe GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=3 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=vcu1525
make run_hw GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=3 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=vcu1525 GEN_BIN_PROGRAM=" fcn 512 512 128 512 128 128 128 1 0 1 0 A0 B0 C0 Bias0 fcn 128 128 128 128 128 128 128 1 0 1 0 A1 B1 C1 Bias1" 
popd

make out_host/lib/libgemxhost.so 
