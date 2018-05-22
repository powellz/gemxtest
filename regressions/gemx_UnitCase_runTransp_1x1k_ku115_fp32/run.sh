#!/usr/bin/env bash

make run_hw GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=1 GEMX_runSpmv=0 GEMX_dataType=float GEMX_part=kcu1500 GEN_BIN_PROGRAM="  transp   32  32   64  96 rm cm  T0 T1  transp 512 768 1024 1152 rm cm  T2 T3" || exit 1
