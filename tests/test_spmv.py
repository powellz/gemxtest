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

xclbin_opts = [] # config data read from config_info.dat

def common_spmv(row,col,data,m,k,nnz,vector_range):
  ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
  if xclbin_opts["GEMX_dataType"] == "float":
    dtype = np.float32
  elif xclbin_opts["GEMX_dataType"] == "int32_t":
    dtype = np.int32
  else:
     raise TypeError("type", xclbin_opts["GEMX_dataType"], "not supported") 
  if dtype == np.int32:
     B = np.random.randint(low=-vector_range, high=vector_range, size=(k, 1), dtype=np.int32)
     C = np.zeros ((m, 1), dtype=np.int32)
     A = gemx.sendSpMat(row,col,data,ddrWidth,dtype)
     gemx.sendMat(B)
     gemx.sendMat(C)
     gemx.addSPMVOp(A,B,C,nnz)
     gemx.execute()
     gemx.getMat(C)
     test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)
  elif dtype == np.float32:
     B = np.zeros ((k, 1), dtype=np.float32)
     test.fillMod(B,k,vector_range)
     C = np.zeros ((m, 1), dtype=np.float32)
     A = gemx.sendSpMat(row,col,data,ddrWidth,dtype)
     gemx.sendMat(B)
     gemx.sendMat(C)
     gemx.addSPMVOp(A,B,C,nnz)
     gemx.execute()
     gemx.getMat(C)
     test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)

def test_spmv_mtxfile(mtxpath,vector_range):
  matA = sio.mmread(mtxpath)
  if sp.issparse(matA):
     row = (matA.row).astype(np.int32)
     col = (matA.col).astype(np.int32)
     data = (matA.data).astype(np.float32)
     m,k = matA.shape
     nnz = matA.nnz    
     # pad with 0s and adjust dimensions when necessary
     ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
     spmvUramGroups = int(xclbin_opts["GEMX_spmvUramGroups"])
     while nnz%ddrWidth !=0:
       row = (np.append(row,0)).astype(np.int32)
       col = (np.append(col,0)).astype(np.int32)
       data = (np.append(data,0)).astype(np.float32)
       nnz = nnz + 1
     while m%(spmvUramGroups*ddrWidth) !=0:  # 16*6 =GEMX_ddrWidth * GEMX_spmvUramGroups
       m = m + 1
     while k%ddrWidth !=0:
       k = k + 1
     print ("size:",m,k,"nnz:",nnz)
     common_spmv(row,col,data,m,k,nnz,vector_range)
  else:
     print ("only sparse matrix is supported")

def test_spmv(m,k,nnz,vector_range):
  row  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
  col  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
  data = np.zeros ((nnz, 1), dtype=np.float32)
  nnz_min = random.randint(-vector_range, vector_range)
  for i in range(nnz):
     nnz_min += 0.3
     data[i,0] = nnz_min
  # pad with 0s and adjust dimensions when necessary
  ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
  spmvUramGroups = int(xclbin_opts["GEMX_spmvUramGroups"])
  while nnz%ddrWidth !=0:
     row = (np.append(row,0)).astype(np.int32)
     col = (np.append(col,0)).astype(np.int32)
     data = (np.append(data,0)).astype(np.float32)
     nnz = nnz + 1
  while m%(spmvUramGroups*ddrWidth) !=0: 
     m = m + 1
  while k%ddrWidth !=0:
     k = k + 1
  print ("size:",m,k,"nnz:",nnz)
  common_spmv(row,col,data,m,k,nnz,vector_range)  

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = SpmvTest()
  args, xclbin_opts = gemx.processCommandLine()
  gemx.createSPMVHandle(args, xclbin_opts)

  #mtx file must be in Matrix Market format
  test_spmv_mtxfile("./data/spmv/mario001.mtx",32764) 
  test_spmv_mtxfile("./data/spmv/image_interp.mtx",32764) 
  #test_spmv_mtxfile("./data/spmv/raefsky3.mtx",32764) 
  #test_spmv_mtxfile("./data/spmv/stomach.mtx",32764)  
  #test_spmv_mtxfile("./data/spmv/torso3.mtx",32764)  
  
  test_spmv(96,128,256,32764)
  test_spmv(65472,65472,500000,32764) 
  test_spmv(12800,12800,1400000,32764) 
  gemx.printStats()
  