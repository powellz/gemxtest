make clean
make run_hw GEMX_ddrWidth=32 GEMX_XddrWidth=16 GEMX_keepMacBits=1 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=1 GEMX_runTransp=0 GEMX_runSpmv=0 GEMX_gemmMBlocks=4 GEMX_gemmKBlocks=4 GEMX_gemmNBlocks=4 GEMX_splitMesh=1 GEMX_part=vcu1525
yes | cp -rf out_hw/gemx.xclbin xclbins/gemm_perf_vcu1525.xclbin
yes | cp -rf out_hw/config_info.dat xclbins/gemm_python_cfg.dat
make clean
make run_hw GEMX_ddrWidth=16 GEMX_argInstrWidth=1 GEMX_numKernels=1 GEMX_runGemv=0 GEMX_runGemm=0 GEMX_runTransp=0 GEMX_runSpmv=1 GEMX_dataType=float GEMX_part=vcu1525
yes | cp -rf out_hw/gemx.xclbin xclbins/spmv_perf_vcu1525.xclbin
