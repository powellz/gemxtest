#!/bin/env bash

##################  Performance reporting  ##################
#gemx multiInstr API V1 test
nice out_host/gemx_api_fcn_multiInstr.exe out_hw/gemx.xclbin 512 512 32 512 32 32 32 1 0 1 0 A0 B0 C0 Bias0 512 512 32 512 32 32 32 1 0 1 0 A1 C0 C1 Bias1 2048 512 32 512 32 32 32 1 0 1 0 A2 C1 C2 Bias2 256 2048 32 2048 32 32 32 1 0 1 0 A3 C2 C3 Bias3 2048 256 32 256 32 32 32 1 0 1 0 A4 C3 C4 Bias4 512 2048 32 2048 32 32 32 1 0 1 0 A5 C4 C5 Bias5 512 512 32 512 32 32 32 1 0 1 0  A6 C5 C6 Bias6  512 512 32 512 32 32 32 1 0 1 0 A7 C6 C7 Bias7 | tee log-multiInstr.txt

nice out_host/gemx_api_fcn_multiInstr.exe out_hw/gemx.xclbin 0 0 0 ../data/fcn/ins.txt ../data/fcn/A0.txt ../data/fcn/B0.txt ../data/fcn/X0.txt 0 0 0 ../data/fcn/ins.txt ../data/fcn/A1.txt C0 ../data/fcn/X1.txt 0 0 0 ../data/fcn/ins.txt A0 C1 X0 | tee log-multiInstr-multiFiles.txt
logs="log-multiInstr.txt log-multiInstr-multiFiles.txt"

#gemx host read input from files test
n=0
while [ $n -le 3 ]; do
  nice out_host/gemx_gen_bin.exe -write out_host/app$n.bin fcn 0 0 0 ../data/fcn/NK$n.txt >> log-NK$n.txt
  nice out_host/gemx_host.exe out_hw/gemx.xclbin out_host/app$n.bin out_hw/app_outNK$n.bin >> log-NK$n.txt 
  nice out_host/gemx_gen_bin.exe -read out_hw/app_outNK$n\0.bin > out_hw/app_outNK$n.txt
  head -22 out_hw/app_outNK$n.txt >> log-NK$n.txt
  out_host/gemx_gen_bin.exe -compare 1e-3 3e-6  out_host/app$n\_gold.bin out_hw/app_outNK$n\0.bin >> log-NK$n.txt
  n=`expr $n \+ 1`
done

nice out_host/gemx_gen_bin.exe -write out_host/app-test.bin fcn 0 0 0 ../data/fcn/ins.txt ../data/fcn/A0.txt ../data/fcn/B0.txt ../data/fcn/X0.txt fcn 0 0 0 ../data/fcn/ins.txt ../data/fcn/A1.txt C0 ../data/fcn/X1.txt fcn 0 0 0 ../data/fcn/ins.txt A0 C1 X0 >> log-test.txt
nice out_host/gemx_host.exe out_hw/gemx.xclbin out_host/app-test.bin out_hw/app_out-test.bin >> log-test.txt 
nice out_host/gemx_gen_bin.exe -read out_hw/app_out-test0.bin > out_hw/app_out-test.txt
head -22 out_hw/app_out-test.txt >> log-test.txt
out_host/gemx_gen_bin.exe -compare 1e-3 3e-6  out_host/app-test_gold.bin out_hw/app_out-test0.bin >> log-test.txt

#python test
export PYTHONPATH=../src/python
python ../tests/test_fcn.py --xclbin out_hw/gemx.xclbin --gemxlib ../out_host/lib/libgemxhost.so --cfg out_hw/config_info.dat | tee log-python.txt

/bin/rm -f perf_gemm_api.csv
egrep -h ^DATA_CSV $logs | grep DdrWidth | head -1 > perf_gemm_api.csv
egrep -h ^DATA_CSV $logs | grep -v DdrWidth >> perf_gemm_api.csv
echo Results are in perf_gemm_api.csv

/bin/rm -f perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep DdrWidth | head -1 > perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep -v DdrWidth >> perf_gemm_api_python.csv