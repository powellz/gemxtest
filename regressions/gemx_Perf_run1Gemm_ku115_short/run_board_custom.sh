#!/bin/env bash

##################  Performance reporting  ##################
ddrWidth=32
n=256

logs=()
while [  $n -le 8192 ]; do
  k=$n
  if [ $k == $ddrWidth ]; then
    k=`expr $k \* 2`
  fi
  echo
  date
  echo "#############  $n ################"
  nice out_host/gemx_api_gemm.exe out_hw/gemx.xclbin  $n $k $n $n $k $n $n 1 0 A B C X | tee log-$n.txt
  logs="$logs log-$n.txt"
  n=`expr $n \* 2`
done

nice out_host/gemx_api_gemm.exe out_hw/gemx.xclbin  2048 27648 2048 27648 2048 2048 2048 1 0 A B C X | tee log-204827648.txt

#python test
export PYTHONPATH=./src/python
python tests/test_gemm.py --xclbin out_hw/gemx.xclbin --gemxlib out_host/lib/libgemxhost.so --cfg out_hw/config_info.dat | tee log-python.txt

/bin/rm -f perf_gemm_api_cpp.csv

egrep -h ^DATA_CSV $logs | grep DdrWidth | head -1 > perf_gemm_api_cpp.csv
egrep -h ^DATA_CSV $logs | grep -v DdrWidth >> perf_gemm_api_cpp.csv

/bin/rm -f perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep DdrWidth | head -1 > perf_gemm_api_python.csv
egrep -h ^DATA_CSV log-python.txt | grep -v DdrWidth >> perf_gemm_api_python.csv
