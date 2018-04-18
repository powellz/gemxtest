import numpy as np
import gemx
import sys
import random
import argparse
import time

# test.py includes all the common test function shared by gemm and fcn engine
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
        
        
  def test_basic_randint (self,A_range, B_range, bias_range, m, k, n, post_scale):
      mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)

      mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)  
      bias = []
      if bias_range != 0:
          bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
      else:
          bias = np.zeros ((m, n), dtype=np.int32, order='C');      
      self.test_basic(mat_A, mat_B, bias, post_scale)

  def test_basic_randint_shift (self,A_range, A_shift, B_range, B_shift, bias_range, bias_shift, m, k, n, post_scale):
      mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
      mat_A = mat_A + A_shift
      mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)
      mat_B = mat_B + B_shift   
      bias = []
      if bias_range != 0:
          bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
      else:
          bias = np.zeros ((m, n), dtype=np.int32);    bias = bias + bias_shift
      self.test_basic(mat_A, mat_B, bias, post_scale)  
      
  def test_rand_basic (self,int_range, bias_range, num_iter, post_scale):  
      min_sz_exp = 8 
      for i in range(num_iter):
          print ("test_rand_basic iter: %d" % i)
          rand_m = random.randint(0, 5) 
          rand_k = random.randint(0, 5) 
          rand_n = random.randint(0, 5)       
          rand_m = 2 ** (rand_m + min_sz_exp) 
          rand_k = 2 ** (rand_k + min_sz_exp)
          rand_n = 2 ** (rand_n + min_sz_exp)
          self.test_basic_randint(int_range, int_range, bias_range, rand_m, rand_k, rand_n, post_scale)
          
  def test_basic(self,mat_A, mat_B, bias, post_scale = [1,1]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test_basic: %d %d %d %d %d" % (m, k, n, post_scale[0], post_scale[1] )) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros( (m, n), dtype=np.int16)
      gemx.sendMat(mat_A)
      gemx.sendMat(mat_B)
      gemx.sendMat(C_fpga)    
      gemx.sendMat(bias)
      gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1]) # default test_basic will call addGEMMOp
      gemx.execute()
      gemx.getMat(C_fpga)  
      if m > 4096 and n > 4096 and k > 4096:
        print("Skip golden comparision because large matrix size")
        else:
        self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
        
class FcnTest(Test):       
  def test_basic(self,mat_A, mat_B, bias, post_scale=[1, 1]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test_basic: %d %d %d %d %d" % (m, k, n, post_scale[0], post_scale[1])) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
      gemx.sendMat(mat_A)
      gemx.sendMat(mat_B)
      gemx.sendMat(C_fpga)    
      gemx.sendMat(bias)
      gemx.addFCNOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], 1, 0)
      gemx.execute()
      gemx.getMat(C_fpga)  
      if m > 4096 and n > 4096 and k > 4096:
        print("Skip golden comparision because large matrix size")
      else:
        self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
        
class GemmTest(Test):               
  pass