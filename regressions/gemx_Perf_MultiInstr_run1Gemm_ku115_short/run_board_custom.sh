#!/bin/bash

##################  Performance reporting  ##################

size=(256 512 1024 2048)
index=0
logs_cpp=()

ddrWidth=32
n=256
while [  $n -le 8192 ]; do
  k=$n
  if [ $k == $ddrWidth ]; then
    k=`expr $k \* 2`
  fi
  echo
  date
  echo "#############  $n ################"
  nice out_host/gemx_api_gemm.exe out_hw/gemx.xclbin  $n $k $n $n $k $n $n 1 0 A B C X | tee log-$n.txt
  logs_cpp="$logs_cpp log-$n.txt"
  n=`expr $n \* 2`
done

#gemx host read input from files test
nice out_host/gemx_gen_bin.exe -write out_host/app-test.bin gemm 0 0 0 data/gemm/ins.txt data/gemm/A0.txt data/gemm/B0.txt data/gemm/X0.txt >> log-test.txt
nice out_host/gemx_host.exe out_hw/gemx.xclbin out_host/app-test.bin out_hw/app_out-test.bin >> log-test.txt 
nice out_host/gemx_gen_bin.exe -read out_hw/app_out-test0.bin > out_hw/app_out-test.txt
head -22 out_hw/app_out-test.txt >> log-test.txt
out_host/gemx_gen_bin.exe -compare 1e-3 3e-6  out_host/app-test_gold.bin out_hw/app_out-test0.bin >> log-test.txt

#compare with the first line in perf_gemm_api_python.csv
nice out_host/gemx_api_gemm_multiInstr.exe out_hw/gemx.xclbin  512 384 128 384 128 128 128 1 0 A B C Bias0 512 512 128 512 128 128 128 1 0 D C E Bias1 2048 512 128 512 128 128 128 1 0 F E G Bias2  128 2048 128 2048 128 128 128 1 0 H G I Bias3 | tee log-multiInstr.txt

#python test
export PYTHONPATH=./src/python
python tests/test_gemm.py --xclbin out_hw/gemx.xclbin --gemxlib out_host/lib/libgemxhost.so --cfg out_hw/config_info.dat | tee log-python.txt

/bin/rm -f perf_gemm_api_cpp.csv
egrep -h ^DATA_CSV $logs_cpp | grep DdrWidth | head -1 > perf_gemm_api_cpp.csv
egrep -h ^DATA_CSV $logs_cpp | grep -v DdrWidth >> perf_gemm_api_cpp.csv

/bin/rm -f perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep DdrWidth | head -1 > perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep -v DdrWidth >> perf_gemm_api_python.csv
