from ctypes import *
import timeit
import numpy as np
import sys
import argparse
class GEMXManager:
  def __init__(self, libFile):
    #self._handle = None
    
    self._lib = cdll.LoadLibrary(libFile)
    self._lib.MakeFCNHost.argtypes = [c_char_p, c_char_p, c_char_p]
    self._lib.MakeGEMMHost.argtypes = [c_char_p, c_char_p, c_char_p]
                
    self._lib.SendToFPGAShrt.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_ulonglong, c_bool]
    self._lib.SendToFPGAInt.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), c_ulonglong, c_bool]
    
    self._lib.AddFCNOp.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"),  
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), 
                                   c_uint, c_uint, c_uint, c_int, c_int, c_short, c_short]
                                   
    self._lib.AddGEMMOp.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"),  
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), 
                                   c_uint, c_uint, c_uint, c_int, c_int]
    self._lib.AddFCNOp.restype = c_bool
    self._lib.AddGEMMOp.restype = c_bool
    
    self._lib.Execute.argtypes = [c_bool]
    self._lib.GetFromFPGA.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_bool]
    self._lib.GetFromFPGA.restype = c_void_p
    self._lib.Wait.argtypes = []
    self._lib.PrintStats.argtypes = []    
    self._lib.GetFreq.argtypes = []  
        
  def createFCNHandle (self, xclbin, kernel, deviceName, numHandles):
     b_xclbin = xclbin.encode('utf-8')
     b_kernel = kernel.encode('utf-8')
     b_device = deviceName.encode('utf-8')
     self._lib.MakeFCNHost(b_xclbin, b_kernel, b_device)
     
  def createGEMMHandle (self, xclbin, kernel, deviceName, numHandles):
     b_xclbin = xclbin.encode('utf-8')
     b_kernel = kernel.encode('utf-8')
     b_device = deviceName.encode('utf-8')
     self._lib.MakeGEMMHost(b_xclbin, b_kernel, b_device)
     
  def addFCNOp(self, A, B, C, bias, postScale, postShift, PReLUScale, PReLUAlpha):
    return self._lib.AddFCNOp(A,B, C, bias, c_uint(A.shape[0]), c_uint( A.shape[1] ), c_uint( B.shape[1]), c_int(postScale), c_int(postShift), c_short(PReLUScale), c_short(PReLUAlpha))
  
  def addGEMMOp(self, A, B, C, bias, postScale, postShift):
    return self._lib.AddGEMMOp(A,B, C, bias, c_uint(A.shape[0]), c_uint( A.shape[1] ), c_uint( B.shape[1]), c_int(postScale), c_int(postShift))
    
  def execute(self, sync_exec = True):
    self._lib.Execute(sync_exec)

  def wait(self):
    self._lib.Wait()    
          
  def sendMat ( self, A,sync_send = False):
    if A.flags['C_CONTIGUOUS'] == False:
        A = np.ascontiguousarray(A)
        print ("Warning: not C_CONTIGUOUS, performance will be affected")
        
    if A.dtype == np.int32:
        self._lib.SendToFPGAInt( A, c_ulonglong(A.size),  sync_send )
    elif A.dtype == np.int16:
        self._lib.SendToFPGAShrt( A, c_ulonglong(A.size),  sync_send )        
    else:
        print ("sendMat: ", A.dtype, " type not supported")
        sys.exit()
    #ctypes.data_as(POINTER(c_float))
    
  def getMat(self, A, sync_get = True):
    self._lib.GetFromFPGA( A, sync_get )
    
  def printStats(self):
    self._lib.PrintStats()
    
  def getFreq(self):
    return self._lib.GetFreq()

_gemxManager = None

def sendMat ( A,sync_send=False):
    _gemxManager.sendMat(A,sync_send)

def getMat (A, sync_get = True):
    return _gemxManager.getMat(A, sync_get)
    
def addFCNOp( A,B,C, bias, postScale, postShift, PReLUScale, PReLUAlpha):
    _gemxManager.addFCNOp(A, B, C, bias, postScale, postShift, PReLUScale, PReLUAlpha)
    
def addGEMMOp( A,B,C, bias, postScale, postShift):
    _gemxManager.addGEMMOp(A, B, C, bias, postScale, postShift)
    
def execute(sync_exec = True):
    _gemxManager.execute( sync_exec)

def wait():
    _gemxManager.wait()    
    
def matmul ( A, B, SendA = True, SendB = True):
    bias = np.zeros ( (A.shape[0], B.shape[1]), dtype=np.int32)   
    return _gemxManager.matmul_addbias(A, B, bias, SendA, SendB,True)

def matmul_addbias ( A, B, bias, SendA = True, SendB = True, SendBias = True):
    return _gemxManager.matmul_addbias(A, B, bias, SendA, SendB, SendBias)

def createManager ( libFile ):
  global _gemxManager
  if not _gemxManager:
    _gemxManager = GEMXManager(libFile)    
  return True  
    
def createFCNHandle(xclbin, kernel, libFile, deviceName, numHandles=1):
  createManager (libFile)
  return _gemxManager.createFCNHandle(xclbin, kernel, deviceName, numHandles)

def createGEMMHandle(xclbin, kernel, libFile, deviceName, numHandles=1):
  createManager (libFile)
  return _gemxManager.createGEMMHandle(xclbin, kernel, deviceName, numHandles)

def printStats():
  return _gemxManager.printStats()
  
def getFreq():
  return _gemxManager.getFreq()

def processCommandLine():
  parser = argparse.ArgumentParser(description='GEMX')
  parser.add_argument('--xclbin', required = True, help='file path to FPGA bitstream')
  parser.add_argument('--gemxlib', required = True, help='file path to GEMX host code shared library')
  parser.add_argument('--device', required=True, choices=['ku115','kcu1500','vu9p', 'vu9pf1'], help='supported FPGA devices')
  return parser

