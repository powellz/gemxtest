#!/bin/env bash

##################  Performance reporting  ##################

logs_cpp=()

nice out_host/gemx_api_gemm_multiInstr.exe out_hw/gemx.xclbin  512 384 128 384 128 128 128 1 0 A B C Bias0 512 512 128 512 128 128 128 1 0 D C E Bias1 2048 512 128 512 128 128 128 1 0 F E G Bias2  128 2048 128 2048 128 128 128 1 0 H G I Bias3 | tee log-multiInstr.txt

logs_cpp="$logs_cpp log-multiInstr.txt"

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

#python test
export PYTHONPATH=./src/python
python tests/test_gemm.py --xclbin out_hw/gemx.xclbin --gemxlib out_host/lib/libgemxhost.so --cfg out_hw/config_info.dat | tee log-python.txt

/bin/rm -f perf_gemm_api_cpp.csv
egrep -h ^DATA_CSV $logs_cpp | grep DdrWidth | head -1 > perf_gemm_api_cpp.csv
egrep -h ^DATA_CSV $logs_cpp | grep -v DdrWidth >> perf_gemm_api_cpp.csv

/bin/rm -f perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep DdrWidth | head -1 > perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep -v DdrWidth >> perf_gemm_api_python.csv
