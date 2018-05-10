#!/usr/bin/env bash

make run_hw GEMX_ddrWidth=16 GEMX_XddrWidth=16 GEMX_keepMacBits=0 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=ku115 GEMX_dataType=float GEMX_XdataType=int32_t GEN_BIN_PROGRAM="gemm 512 512 512  512 512 512 512 1 0 A05 B05 C05 X05  gemm 1024 1024 1024  1024 1024 1024 1024 1 0 A1k B1k C1k X1K   gemm 1024 1024 1024  1536 2048 2560 1024 1 0 A1kld B1kld C1kld X1kld" || exit 1

# make run_hw GEMX_ddrWidth=16 GEMX_XddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=ku115 GEMX_dataType=float GEMX_XdataType=int32_t GEN_BIN_PROGRAM="gemm 256 256 256  256 256 256 256 1 0 A05 B05 C05 X05"

#make run_hw GEMX_ddrWidth=16 GEMX_XddrWidth=8 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=ku115 GEMX_dataType=float GEMX_XdataType=int64_t GEN_BIN_PROGRAM="gemm 256 256 256  256 256 256 256 1 0 A05 B05 C05 X05"
