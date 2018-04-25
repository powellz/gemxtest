import numpy as np
import gemx
import sys
import random
import argparse
import time
import test
from test import SpmvTest
#from scipy.io import mmread
#from scipy.sparse import coo_matrix
  
def test_spmv(m,k,nnz,nnz_range,dtype):
  row  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
  col  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
  dataint = np.random.randint(low=1, high=nnz_range, size=(nnz, 1), dtype=np.int64)
  data = dataint.astype(np.double)
  #A = coo_matrix((data, (row, col)), shape=(m, k))
  if dtype == np.int16:
     B = np.random.randint(low=-nnz_range, high=nnz_range, size=(k, 1), dtype=np.int16)
     C = np.zeros ((m, 1), dtype=np.int16)
  elif dtype == np.int32:
     B = np.random.randint(low=-nnz_range, high=nnz_range, size=(k, 1), dtype=np.int32)
     C = np.zeros ((m, 1), dtype=np.int32)
  elif dtype == np.float32:
     Bint = np.random.randint(low=-nnz_range, high=nnz_range, size=(k, 1), dtype=np.int32)
     B = np.zeros ((k, 1), dtype=np.float32)
     B = Bint.astype(np.float32)
     C = np.zeros ((m, 1), dtype=np.float32)
  else:
     raise TypeError("type", dtype, "not supported")
  gemx.sendSparse(row,col,data,m,k,nnz,B,C,0)
  gemx.execute(0)
  gemx.getMat(C,0)
  test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = SpmvTest()
  parser = gemx.processCommandLine()
  args = parser.parse_args()
  timePoint = []
  timePoint.append(time.time()*1000) #current time
  gemx.createSPMVHandle(args.xclbin, args.gemxlib, args.device, args.numKernel)
  timePoint.append(time.time()*1000) # local xclbin
  print "Load Xclbin Time:",
  print timePoint[1] - timePoint[0]
  test_spmv(96,128,256,2,np.float32)
