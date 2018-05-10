#!/bin/env bash

##################  Performance reporting  ##################
ddrWidth=32
#n=$ddrWidth
n=512

logs=()
while [  $n -lt 16384 ]; do
  k=$n
  if [ $k == $ddrWidth ]; then
    k=`expr $k \* 2`
  fi
  echo
  date
  echo "#############  $n ################"
  #nice gemx_api_gemm.exe gemx.xclbin  $n $k $n | tee log-$n.txt
  #nice ref/out_host/gemx_api_gemm.exe ref/out_hw/gemx.xclbin  $n $k $n | tee log-$n.txt
  nice out_host/gemx_api_gemm.exe out_hw/gemx.xclbin  $n $k $n $n $k $n $n 1 0 A B C X | tee log-$n.txt
  logs="$logs log-$n.txt"
  n=`expr $n \* 2`
done

nice out_host/gemx_api_gemm.exe out_hw/gemx.xclbin  2048 27648 2048 27648 2048 2048 2048 1 0 A B C X  | tee log-27648.txt
logs="$logs log-27648.txt"

/bin/rm -f perf_gemm_api.csv
egrep -h ^DATA_CSV $logs | grep DdrWidth | head -1 > perf_gemm_api.csv
egrep -h ^DATA_CSV $logs | grep -v DdrWidth >> perf_gemm_api.csv
echo Results are in perf_gemm_api.csv

