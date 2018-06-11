#!/bin/env bash

##################  Performance reporting  ##################
#gemx multiInstr API V1 test
nice out_host/gemx_api_pca.exe out_hw/gemx.xclbin 0 0 0 8 ../data/spmv/mario001.mtx A0 B0 C0 38496 38496 114648 8 none A0 C0 C1 0 0 0 8 ../data/spmv/garon2.mtx A2 B2 C2 13536 13536 390608 8 none A2 C2 C3 0 0 0 8 ../data/spmv/Muu.mtx A4 B4 C4 7104 7104 88624 8 none A4 C4 C5 0 0 0 8 ../data/spmv/chem_master1.mtx A6 B6 C6 40416 40416 201208 8 none A6 C6 C7 0 0 0 8 ../data/spmv/c-67.mtx A8 B8 C8 57975 57975 294955 8 none A8 C8 C9 | tee log-multiInstr.txt

#cases that might have mismatches
nice out_host/gemx_gen_bin.exe -write app.bin pca 0 0 0 8 ../data/spmv/dawson5.mtx A0 B0 C0 pca 51552 51552 531160 8 none A0 C0 C1 pca 0 0 0 8 ../data/spmv/bcsstk16.mtx A2 B2 C2 pca 4896 4896 147632 8 none A2 C2 C3 > log-mismatch.txt
nice out_host/gemx_host.exe out_hw/gemx.xclbin app.bin app_out.bin >> log-mismatch.txt
nice out_host/gemx_gen_bin.exe -compare 1e-3 3e-6 app_gold.bin app_out0.bin >> log-mismatch.txt

logs="log-multiInstr.txt"

/bin/rm -f perf_pca_api.csv
egrep -h ^DATA_CSV $logs | grep DdrWidth | head -1 > perf_pca_api.csv
egrep -h ^DATA_CSV $logs | grep -v DdrWidth >> perf_pca_api.csv
echo Results are in perf_pca_api.csv
