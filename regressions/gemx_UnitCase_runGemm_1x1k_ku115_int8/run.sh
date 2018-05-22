#!/usr/bin/env bash

make run_hw GEMX_ddrWidth=32 GEMX_XddrWidth=32 GEMX_keepMacBits=0 GEMX_argInstrWidth=2 GEMX_dataType=int8_t GEMX_dataEqIntType=int8_t GEMX_XdataType=int8_t GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_part=kcu1500 GEN_BIN_PROGRAM="gemm 256 256 256  256 256 256 256 1 0 A25 B25 C25 X25 gemm 512 512 512  512 512 512 512 1 0 A05 B05 C05 X05 gemm 1024 1024 1024  1024 1024 1024 1024 1 0 A1k B1k C1k X1k   gemm 1024 1024 1024  1536 2048 2560 2560 1 0 A1kld B1kld C1kld X1kld  " || exit 1
