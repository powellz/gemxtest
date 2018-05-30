#!/usr/bin/env bash
make out_host/gemx_api_gemm.exe GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=kcu1500
make out_host/gemx_api_gemm_multiInstr.exe GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=kcu1500
make gemm_test_python GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=1 GEMX_splitMesh=1 GEMX_part=kcu1500 GEN_BIN_PROGRAM="gemm 512 512 32 512 32 32 32 1 0 A B C Bias0 gemm 512 512 32 512 32 32 32 1 0 D C E Bias1 gemm 2048 512 32 512 32 32 32 1 0 F E G Bias2 gemm 256 2048 32 2048 32 32 32 1 0 H G I Bias3 gemm 2048 256 32 256 32 32 32 1 0 J I K Bias4 gemm 512 2048 32 2048 32 32 32 1 0 L K M Bias5 gemm 512 512 32 512 32 32 32 1 0 N M O Bias6 gemm 512 512 32 512 32 32 32 1 0 P O Q Bias7" || exit 1
