/tools/batonroot/rodin/devkits/lnx64/gcc-6.2.0/bin/g++ -g -O2 -std=c++11 -D GEMX_dataType=short -L/$PWD/out_host -lxblas -Wl,--rpath=/$PWD/out_host src/gemx_xblas_main.cpp -o gemx_xblas_main.exe 
export GEMX_XCLBIN_PATH=out_hw/gemx.xclbin
nice gemx_xblas_main.exe  | tee log-xblas.txt

/bin/rm -f perf_gemm_api.csv
egrep -h ^DATA_CSV log-xblas.txt | grep DdrWidth | head -1 > perf_gemm_api.csv
egrep -h ^DATA_CSV log-xblas.txt | grep -v DdrWidth >> perf_gemm_api.csv
echo Results are in perf_gemm_api.csv
