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

make out_host/lib/libgemxhost.so

make out_host/gemx_api_spmv.exe GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=1 GEMX_dataType=float GEMX_dataEqIntType=int32_t GEMX_idxType=int32_t GEMX_nnzBlocks=1 GEMX_useURAM=1 GEMX_part=vcu1525

make run_hw GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=1 GEMX_dataType=float GEMX_dataEqIntType=int32_t GEMX_idxType=int32_t GEMX_nnzBlocks=1 GEMX_useURAM=1 GEMX_part=vcu1525 GEN_BIN_PROGRAM="spmv 96 128 256 none A0 B0 C0  spmv 65472 65472 500000 none A1 B1 C1  spmv 0 0 0 data/spmv/diag16.mtx A2 B2 C2  spmv 0 0 0 data/spmv/diag64.mtx A3 B3 C3  spmv 0 0 0 data/spmv/diag256.mtx A4 B4 C4  spmv 0 0 0 data/spmv/256_1k.mtx A5 B5 C5  spmv 0 0 0 data/spmv/diag1024.mtx A6 B6 C6  spmv 0 0 0 data/spmv/diag16k.mtx.gz A7 B7 C7 spmv 0 0 0 data/spmv/image_interp.mtx A8 B8 C8 spmv 0 0 0 data/spmv/mario001.mtx A9 B9 C9 spmv 0 0 0 data/spmv/dawson5.mtx A10 B10 C10 spmv 0 0 0 data/spmv/bcsstk16.mtx A11 B11 C11" || exit 1
