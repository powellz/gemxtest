#!/usr/bin/env bash

make run_hw GEMX_ddrWidth=32 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=1 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_transpBlocks=8 GEMX_part=kcu1500 GEN_BIN_PROGRAM="gemv 512 512 512  A1 B1 C1 gemv 1024 1024 1024 A2 B2 C2 gemv 2048 2048 2048 A3 B3 C3 gemv 4096 4096 4096 A4 B4 C4 gemv 8192 8192 8192 A5 B5 C5 gemv 16384 16384 16384 A6 B6 C6" || exit 1
