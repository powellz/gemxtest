#!/usr/bin/env bash

make run_hw GEMX_ddrWidth=32 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=1 GEMX_runSpmv=0 GEMX_transpBlocks=8 GEMX_part=ku115 GEN_BIN_PROGRAM="transp 512 512 512 512 rm cm T0 T1 transp 1024 1024 1024 1024 rm cm T2 T3 transp 2048 2048 2048 2048 rm cm  T4 T5 transp 4096 4096 4096 4096 rm cm  T6 T7 transp 8192 8192 8192 8192 rm cm  T8 T9 transp 16384 16384 16384 16384 rm cm  T10 T11" || exit 1
