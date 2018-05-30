#!/bin/env bash

##################  Performance reporting  ##################
#gemx multiInstr API V1 test
nice out_host/gemx_api_pca.exe out_hw/gemx.xclbin 0 0 0 8 ../data/spmv/mario001.mtx A0 B0 C0 38496 38496 114648 8 none A0 C0 C1 | tee log-multiInstr.txt
logs="log-multiInstr.txt"

/bin/rm -f perf_pca_api.csv
egrep -h ^DATA_CSV $logs | grep DdrWidth | head -1 > perf_pca_api.csv
egrep -h ^DATA_CSV $logs | grep -v DdrWidth >> perf_pca_api.csv
echo Results are in perf_pca_api.csv
