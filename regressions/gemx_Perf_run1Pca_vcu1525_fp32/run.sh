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

pushd ./pca
make out_host/gemx_api_pca.exe GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_dataType=float GEMX_dataEqIntType=int32_t GEMX_part=vcu1525
make run_hw GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_dataType=float GEMX_dataEqIntType=int32_t GEMX_part=vcu1525 GEN_BIN_PROGRAM="pca 0 0 0 8 ../data/spmv/mario001.mtx A0 B0 C0 pca 38496 38496 114648 8 none A0 C0 C1 pca 0 0 0 8 ../data/spmv/garon2.mtx A2 B2 C2 pca 13536 13536 390608 8 none A2 C2 C3 pca 0 0 0 8 ../data/spmv/Muu.mtx A4 B4 C4 pca 7104 7104 88624 8 none A4 C4 C5 pca 0 0 0 8 ../data/spmv/chem_master1.mtx A6 B6 C6 pca 40416 40416 201208 8 none A6 C6 C7 pca 0 0 0 8 ../data/spmv/c-67.mtx A8 B8 C8 pca 57984 57984 294960 8 none A8 C8 C9" 
popd
