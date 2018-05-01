import numpy as np
import gemx
import sys
import random
import argparse
import time
import test
import scipy.io as sio
import scipy.sparse as sp
from test import SpmvTest

def common_spmv(row,col,data,m,k,nnz,vector_range,dtype):
  if dtype == np.int32:
     B = np.random.randint(low=-vector_range, high=vector_range, size=(k, 1), dtype=np.int32)
     C = np.zeros ((m, 1), dtype=np.int32)
     gemx.sendSparse(row,col,data,m,k,nnz,B,C,0)
     gemx.executefloat(0)
     gemx.getMat(C,0)
     test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)
  elif dtype == np.float32:
     B = np.zeros ((k, 1), dtype=np.float32)
     fillMod(B,B.shape[0],9)
     C = np.zeros ((m, 1), dtype=np.float32)
     gemx.sendSparse(row,col,data,m,k,nnz,B,C,0)
     gemx.executefloat(0)
     gemx.getMat(C,0)
     test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)
  else:
     raise TypeError("type", dtype, "not supported") 

def fillMod(B,size,Max):
  l_val = 1.0
  l_step = 0.3
  l_drift = 0.00001
  l_sign = 1
  for i in range(size):
     B[i,0] = l_val
     l_val += l_sign * l_step
     l_step += l_drift
     l_sign = -l_sign;
     if l_val > Max:
        l_val -= Max

def test_spmv_mtxfile(mtxpath,vector_range,dtype):
  matA = sio.mmread(mtxpath)
  if sp.issparse(matA):
     row = (matA.row).astype(np.int32)
     col = (matA.col).astype(np.int32)
     data = (matA.data).astype(np.float32)
     m,k = matA.shape
     nnz = matA.nnz
     # pad with 0s and adjust dimensions when necessary
     while nnz%16 !=0:
       row = (np.append(row,0)).astype(np.int32)
       col = (np.append(col,0)).astype(np.int32)
       data = (np.append(data,0)).astype(np.float32)
       nnz = nnz + 1
     while m%96 !=0:  # 16*6 =GEMX_ddrWidth * GEMX_spmvUramGroups
       m = m + 1
     while k%16 !=0:
       k = k + 1
     common_spmv(row,col,data,m,k,nnz,vector_range,dtype)
  else:
     print ("only sparse matrix is supported")
  
def test_spmv(m,k,nnz,nnz_range,dtype):
  row  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
  col  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
  data = np.zeros ((nnz, 1), dtype=np.float32)
  for i in range(nnz):
     nnz_range += 0.3
     data[i,0] = nnz_range
  # pad with 0s and adjust dimensions when necessary
  while nnz%16 !=0:
     row = (np.append(row,0)).astype(np.int32)
     col = (np.append(col,0)).astype(np.int32)
     data = (np.append(data,0)).astype(np.float32)
     nnz = nnz + 1
  while m%96 !=0:  # 16*6 =GEMX_ddrWidth * GEMX_spmvUramGroups
     m = m + 1
  while k%16 !=0:
     k = k + 1
  common_spmv(row,col,data,m,k,nnz,nnz_range,dtype)  

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = SpmvTest()
  parser = gemx.processCommandLine()
  args = parser.parse_args()
  timePoint = []
  timePoint.append(time.time()*1000) #current time
  gemx.createSPMVHandle(args.xclbin, args.gemxlib, args.device, args.numKernel)
  timePoint.append(time.time()*1000) # local xclbin
  print ("Load Xclbin Time:"),
  print (timePoint[1] - timePoint[0])
  
  #test_spmv(96,128,256,2,np.int32)
  #mtx file must be in Matrix Market format
  #test_spmv_mtxfile("/wrk/xsjhdnobkup2/yifei/git_gemx/gemx/gemx/tests/raefsky3.mtx",2,np.float32) # seg error on execute()
  #test_spmv(96,128,256,2,np.float32)
  test_spmv(65472,65472,500000,17,np.float32) 
  #test_spmv(960,960,2000,17,np.float32) 