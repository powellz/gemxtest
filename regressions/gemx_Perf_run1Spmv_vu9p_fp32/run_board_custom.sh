#!/bin/env bash

##################  Performance reporting  ##################

logs=()

echo
date

nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/diag16k.mtx |& tee log-diag16k.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/raefsky3.mtx |& tee log-raefsky3.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/stomach.mtx |& tee log-stomach.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/torso3.mtx |& tee log-torso3.txt
#nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/Rucci1.mtx |& tee log-Rucci1.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 65472 65472 500000 |& tee log-1.txt

#logs="$logs log-diag16k.txt log-raefsky3.txt log-stomach.txt log-torso3.txt log-Rucci1.txt log-1.txt"
logs="$logs log-diag16k.txt log-raefsky3.txt log-stomach.txt log-torso3.txt log-1.txt"
 
/bin/rm -f perf_gemm_api.csv
egrep -h ^DATA_CSV $logs | grep DdrWidth | head -1 > perf_gemm_api.csv
egrep -h ^DATA_CSV $logs | grep -v DdrWidth >> perf_gemm_api.csv
echo Results are in perf_gemm_api.csv
