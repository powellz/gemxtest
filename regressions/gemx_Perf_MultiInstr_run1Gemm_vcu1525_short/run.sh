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

make out_host/gemx_api_gemm_multiInstr.exe GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_ddrWidth=32 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=vcu1525
make gemm_test_python GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=vcu1525 GEN_BIN_PROGRAM=" gemm 512 512 32 512 32 32 32 1 0 A0 B0 C0 Bias0 gemm 512 512 32 512 32 32 32 1 0 A1 C0 C1 Bias1 gemm 2048 512 32 512 32 32 32 1 0 A2 C1 C2 Bias2 gemm 256 2048 32 2048 32 32 32 1 0 A3 C2 C3 Bias3 gemm 2048 256 32 256 32 32 32 1 0 A4 C3 C4 Bias4 gemm 512 2048 32 2048 32 32 32 1 0 A5 C4 C5 Bias5 gemm 512 512 32 512 32 32 32 1 0  A6 C5 C6 Bias6 gemm 512 512 32 512 32 32 32 1 0 A7 C6 C7 Bias7" || exit 1

#make -f gemx_host/Makefile
