import gemx
import numpy as np
import time
import argparse

def matrix_dim(s):
    try:
        #print (s)
        m,k,n = map(int, s.split(','))
        return m,k,n
    except:
        raise argparse.ArgumentTypeError("matrix dims must be a triplet [m,k,n]")

parser = gemx.default_args()
parser.add_argument('--numiter', required = False, type = int, default = 10000)
parser.add_argument('-m','--matrix', action='append', type=matrix_dim, help='m,k,n dimensions of matrices', required=True)
args = parser.parse_args()
xclbin_opts = gemx.parse_cfg ( args.cfg ) 

gemx.createFCNHandle(args, xclbin_opts)

A_buf = []
bias_buf = []
B_buf = []
C_buf = []

num_matrix = len(args.matrix)
for dim_set in args.matrix:
    m = int(dim_set[0])
    k = int(dim_set[1])
    n = int(dim_set[2])
    print (m,k,n)
    A_buf.append(  np.zeros ( (m,k), dtype=np.int16, order='C') )
    bias_buf.append ( np.zeros ( (m,n), dtype=np.int32, order='C') )
    B_buf.append( np.zeros ( (k,n),  dtype=np.int16, order='C'))
    C_buf.append( np.zeros ( (m,n), dtype=np.int16, order='C') )    
        
for i in range( num_matrix ):
    gemx.sendMat(B_buf[i])    
    gemx.sendMat(A_buf[i])    
    gemx.sendMat(C_buf[i])
    gemx.sendMat(bias_buf[i])    

time.sleep(2)
total_time = 0
for k in range( args.numiter):
    start_time = time.time()
    gemx.sendMat(B_buf[0])
    for i in range( num_matrix ):
        #gemx.addFCNOp(A_buf[i], B_buf[i], B_buf[i+1], bias_buf[i], 1,0,1,0 )
        gemx.addFCNOp(A_buf[i], B_buf[i], C_buf[i], bias_buf[i], 1,0,1,0 )
        
    gemx.execute()
    gemx.getMat(C_buf[num_matrix-1])
    #gemx.wait()    
    total_time += time.time() - start_time

print ("Average FPGA exec time(python): ", (total_time/ args.numiter)*1000, " ms")

gemx.printStats()