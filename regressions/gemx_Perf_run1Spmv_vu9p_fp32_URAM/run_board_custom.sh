#!/bin/env bash

##################  Performance reporting  ##################

logs=()

echo
date

nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/diag16k.mtx.gz |& tee log-diag16k.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/image_interp.mtx |& tee log-image_interp.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/mario001.mtx |& tee log-mario001.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/garon2.mtx |& tee log-garon2.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/chem_master1.mtx |& tee log-chem_master1.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/Muu.mtx |& tee log-Muu.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 0 0 0 data/spmv/c-67.mtx |& tee log-c-67.txt
nice out_host/gemx_api_spmv.exe out_hw/gemx.xclbin 65472 65472 500000 |& tee log-1.txt

logs="$logs log-diag16k.txt log-image_interp.txt log-mario001.txt log-garon2.txt log-chem_master1.txt log-Muu.txt log-c-67.txt log-1.txt"

#python test
export PYTHONPATH=./src/python
python -u tests/test_spmv.py --xclbin out_hw/gemx.xclbin --gemxlib out_host/lib/libgemxhost.so --cfg out_hw/config_info.dat >& log-python.txt

/bin/rm -f perf_spmv_api.csv
egrep -h ^DATA_CSV $logs | grep DdrWidth | head -1 > perf_spmv_api.csv
egrep -h ^DATA_CSV $logs | grep -v DdrWidth >> perf_spmv_api.csv
echo Results are in perf_spmv_api.csv
