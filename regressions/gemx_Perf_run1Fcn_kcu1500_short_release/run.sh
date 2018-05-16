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
make out_host/gemx_api_fcn_multiInstr.exe GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=kcu1500
make run_hw GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=kcu1500 GEN_BIN_PROGRAM=" fcn 512 512 32 512 32 32 32 1 0 1 0 A0 B0 C0 Bias0 fcn 512 512 32 512 32 32 32 1 0 1 0 A1 C0 C1 Bias1 fcn 2048 512 32 512 32 32 32 1 0 1 0 A2 C1 C2 Bias2 fcn 256 2048 32 2048 32 32 32 1 0 1 0 A3 C2 C3 Bias3 fcn 2048 256 32 256 32 32 32 1 0 1 0 A4 C3 C4 Bias4 fcn 512 2048 32 2048 32 32 32 1 0 1 0 A5 C4 C5 Bias5 fcn 512 512 32 512 32 32 32 1 0 1 0  A6 C5 C6 Bias6 fcn 512 512 32 512 32 32 32 1 0 1 0 A7 C6 C7 Bias7" 
popd

make out_host/lib/libgemxhost.so 
