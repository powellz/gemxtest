import numpy as np
import gemx
import sys
import random
import argparse
import time

# test.py includes all the common test function shared by gemm, fcn and spmv engine
class Test:
  def cmp(self,A, B):
      if np.array_equal(A, B):
          print ("Success!\n")
      else:
          print ("not equal :(")
          sys.exit()  

  def multiply_and_cmp(self,C, A, B, X, m, n, post_scale):
      # Calculate golden C
      m64 = np.matmul(np.int64(A), np.int64(B))  # intermediate accumulation to 64 bits
      bias64 = np.int64(X)  # bias to 64 bits
      output64 = m64 + bias64
      o64d = output64 * post_scale[0]
      o64m = o64d / (2 ** post_scale[1])
      C_cpu = np.int16(o64m)  # scale down for 16 bits
      if np.array_equal(C, C_cpu):
          print ("Success!\n")
      else:
          print ("Not equal!")
          print (C.shape, C_cpu.shape)
          np.savetxt("cpu_out.np", C_cpu, fmt="%d")
          np.savetxt("fpga_out.np", C, fmt="%d")
          np.savetxt("bias.np", X, fmt="%d")
          np.savetxt("A.np", A, fmt="%d")
          np.savetxt("B.np", B, fmt="%d")
          sys.exit();    
        
        
  def test_basic_randint (self,PE, A_range, B_range, bias_range, m, k, n, post_scale):
      mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
      mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)  
      bias = []
      if bias_range != 0:
          bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
      else:
          bias = np.zeros ((m, n), dtype=np.int32, order='C');      
      self.test_basic(PE,mat_A, mat_B, bias, post_scale)

  def test_basic_randint_shift (self,PE,A_range, A_shift, B_range, B_shift, bias_range, bias_shift, m, k, n, post_scale):
      mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
      mat_A = mat_A + A_shift
      mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)
      mat_B = mat_B + B_shift   
      bias = []
      if bias_range != 0:
          bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
      else:
          bias = np.zeros ((m, n), dtype=np.int32);    bias = bias + bias_shift
      self.test_basic(PE,mat_A, mat_B, bias, post_scale)  
      
  def test_rand_basic (self,PE,int_range, bias_range, num_iter, post_scale):  
      min_sz_exp = 8 
      for i in range(num_iter):
          print ("test_rand_basic iter: %d" % i)
          rand_m = random.randint(0, 5) 
          rand_k = random.randint(0, 5) 
          rand_n = random.randint(0, 5)       
          rand_m = 2 ** (rand_m + min_sz_exp) 
          rand_k = 2 ** (rand_k + min_sz_exp)
          rand_n = 2 ** (rand_n + min_sz_exp)
          self.test_basic_randint(PE,int_range, int_range, bias_range, rand_m, rand_k, rand_n, post_scale)
          
  def test_basic(self,PE, mat_A, mat_B, bias, post_scale = [1,1]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test_basic(PE=%d): %d %d %d %d %d" % (PE,m, k, n, post_scale[0], post_scale[1] )) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros( (m, n), dtype=np.int16)
      gemx.sendMat(mat_A,PE)
      gemx.sendMat(mat_B,PE)
      gemx.sendMat(C_fpga,PE)    
      gemx.sendMat(bias, PE)
      gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], PE) # default test_basic will call addGEMMOp
      gemx.execute(PE)
      gemx.getMat(C_fpga,PE)  
      if m > 4096 and n > 4096 and k > 4096:
        print("Skip golden comparision because large matrix size")
      else:
        self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
   
  def test_perf(self,timePointKernel, total_operations, total_parallel_operations, freq, m, k, n):
      Execute_Time = (timePointKernel[2] - timePointKernel[1])*1e3
      API_Time = (timePointKernel[3] - timePointKernel[0])*1e3
      timeMsAt100pctEff = total_parallel_operations / 2 / 32 / 32 / ( freq * 1e6 ) * 1e3
      effKernelPct = 100 * timeMsAt100pctEff / Execute_Time
      effApiPct = 100 * timeMsAt100pctEff / API_Time
      perfKernelInTops = total_operations / (Execute_Time * 1e-3) / 1e12
      perfApiInTops = total_operations/ (API_Time * 1e-3) / 1e12;
      print ("DATA_CSV:DdrWidth,Freq,M,K,N,Ops,TimeKernelMs,TimeApiMs,EffKernelPct,EffApiPct,PerfKernelTops,PerfApiTops")
      print ("DATA_CSV:32,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f" % (freq,m,k,n,total_operations,Execute_Time,API_Time,effKernelPct,effApiPct,perfKernelInTops,perfApiInTops))
  
  def check_input(self, m_size, k_size, n_size, xclbin_opts):
      m_block = int(xclbin_opts["GEMX_gemmMBlocks"])
      k_block = int(xclbin_opts["GEMX_gemmKBlocks"])
      n_block = int(xclbin_opts["GEMX_gemmNBlocks"])
      ddr_width = int(xclbin_opts["GEMX_ddrWidth"])
      if m_size%(m_block*ddr_width) !=0:
         print ("m must be multiple of", m_block, "and", ddr_width)
         sys.exit()
      elif k_size%(k_block*ddr_width) !=0:
         print ("k must be multiple of", k_block, "and", ddr_width)
         sys.exit()
      elif n_size%(n_block*ddr_width) !=0:
         print ("n must be multiple of", n_block, "and", ddr_width)  
         sys.exit()
        
class FcnTest(Test):       
  def test_basic(self,PE, mat_A, mat_B, bias, post_scale=[1, 1]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test_basic: %d %d %d %d %d" % (m, k, n, post_scale[0], post_scale[1])) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
      gemx.sendMat(mat_A, PE)
      gemx.sendMat(mat_B, PE)
      gemx.sendMat(C_fpga, PE)    
      gemx.sendMat(bias, PE)
      gemx.addFCNOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], 1, 0, PE)
      gemx.execute(PE)
      gemx.getMat(C_fpga, PE)  
      if m > 4096 and n > 4096 and k > 4096:
        print("Skip golden comparision because large matrix size")
      else:
        self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
        
class SpmvTest(Test):
  def multiply_and_cmp_spmv(self,row,col,data,m,k,nnz,B,C):
      if B.dtype == np.int32:
        C_cpu = np.zeros ((m, 1), dtype=np.int32)
        data_cpu = np.zeros ((m, 1), dtype=np.int32)
        data_cpu = data.astype(np.int32)
      elif B.dtype == np.float32:
        C_cpu = np.zeros ((m, 1), dtype=np.float32)
        data_cpu = np.zeros ((m, 1), dtype=np.float32)
        data_cpu = data.astype(np.float32)
      else:
        raise TypeError("type", B.dtype, "not supported") 
      for i in range(nnz):
        C_cpu[row[i]] += B[col[i]] * data_cpu[i]
      print (C.shape, C_cpu.shape)
      np.savetxt("C.np", C, fmt="%f")
      np.savetxt("C_cpu.np", C_cpu, fmt="%f")
      self.cmp(C, C_cpu)

        
class GemmTest(Test):               
  pass