#!/usr/bin/env bash

make run_hw GEMX_ddrWidth=64 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=1 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_dataType=int8_t GEMX_dataEqIntType=int8_t  GEMX_part=kcu1500 GEN_BIN_PROGRAM="gemv  256 256 320 A1 B1 C1  gemv  512 768 1024 A5 B5 C5" || exit 1