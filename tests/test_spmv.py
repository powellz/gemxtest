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
     Bint = np.random.randint(low=-vector_range, high=vector_range, size=(k, 1), dtype=np.int32)
     B = np.zeros ((k, 1), dtype=np.float32)
     B = Bint.astype(np.float32)
     C = np.zeros ((m, 1), dtype=np.float32)
     gemx.sendSparse(row,col,data,m,k,nnz,B,C,0)
     gemx.executefloat(0)
     gemx.getMat(C,0)
     test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)
  else:
     raise TypeError("type", dtype, "not supported") 

def test_spmv_mtxfile(mtxpath,vector_range,dtype):
  matA = sio.mmread(mtxpath)
  if sp.issparse(matA):
     row = (matA.row).astype(np.int32)
     col = (matA.col).astype(np.int32)
     data = (matA.data).astype(np.float32)
     m,k = matA.shape
     nnz = matA.nnz
     common_spmv(row,col,data,m,k,nnz,vector_range,dtype)
  else:
     print ("only sparse matrix is supported")
  
def test_spmv(m,k,nnz,nnz_range,dtype):
  row  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
  col  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
  dataint = np.random.randint(low=1, high=nnz_range, size=(nnz, 1), dtype=np.int32)
  data = dataint.astype(np.float32)
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
  #test_spmv_mtxfile(PATH_TO_MTX_FILE,2,np.float32) # mtx file must be in Matrix Market format
  test_spmv(96,128,256,2,np.float32)
  #test_spmv(65742,65742,500000,2,np.float32) # several mismatches
