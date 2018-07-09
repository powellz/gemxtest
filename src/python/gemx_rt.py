import gemx
import numpy as np
import math

class GemxRT():
    def __init__(self, xclbin_opt, wgt,bias, wgt_scale, post_scale):
        self.min_m = 32*int(xclbin_opt["GEMX_gemmMBlocks"])
        self.min_k = 32*int(xclbin_opt["GEMX_gemmKBlocks"])
        self.min_n = 32*int(xclbin_opt["GEMX_gemmNBlocks"]) 
        if type (wgt) != list:
            wgt = [wgt]
        
        if type(bias) != list:
            bias = [bias]
            
        self._wshape = []
        for w in wgt:
            self._wshape.append(w.shape)
        self._qw = [ np.int16(a*b) for a,b in zip(wgt, wgt_scale)]
        self._qb = [ np.int32(a*b) for a,b in zip(bias, wgt_scale)]
        self._qw = self.format_for_fpga( self._qw, self.min_k, self.min_n)
        self._qb = self.format_for_fpga( self._qb, self.min_m, self.min_n)
        gemx.load_buf( self._qw )
        gemx.load_buf( self._qb )
        #in_row, in_col = self.get_padded_shape(in_dim, self.min_m, self.min_k)
        self.fpga_buf = []
        self.out_dim = None
        self.post_scale = post_scale
        
    def get_padded_shape ( self, shape, min_row, min_col):
        row_padded = int( math.ceil( np.float32(shape[0]) / min_row ) * min_row ) 
        col_padded = int( math.ceil( np.float32(shape[1]) / min_col ) * min_col )
        return row_padded,col_padded

    def format_for_fpga ( self, np_list, min_row, min_col):
        padded_list = []
        for m in np_list:
            if m.ndim == 1:
                m = m.reshape(m.shape[0],1)
    
            row_padded, col_padded = self.get_padded_shape ( m.shape, min_row, min_col)
            padded_arr = np.zeros ( (row_padded, col_padded), dtype=m.dtype, order='C')
            padded_arr[0:m.shape[0], 0:m.shape[1]] = m
#           print ("padded shape", padded_arr.shape)  
#           print (padded_arr)            
            padded_list.append(padded_arr)
        return padded_list
    
    def init_databuf (self, in_shape ):
        row_padded, col_padded =  self.get_padded_shape( in_shape, self.min_m, self.min_k)
        if not self.fpga_buf:       
            self.fpga_buf = self.create_databuf( self._qw, [row_padded, col_padded])
            self.out_dim = in_shape
            for i in self._wshape:
                self.out_dim = ( self.out_dim[0], i[1] )
        
        return row_padded, col_padded
        
        
    def create_databuf ( self, q_wt, inp_shape):
        fpga_buf = []
        buf_shape = inp_shape
        fpga_buf.append ( gemx.create_fpga_buf( buf_shape, q_wt[0].dtype ) )
        for w in q_wt:
            buf_shape = ( buf_shape[0], w.shape[1] )
            fpga_buf.append ( gemx.create_fpga_buf( buf_shape, w.dtype ) )
        
        return fpga_buf             
    
    def loadInstr(self):
        gemx.clearInstrBuf()
        for i,(w_i,b_i) in enumerate( zip( self._qw, self._qb) ):
            gemx.addGEMMOp( self.fpga_buf[i], w_i , self.fpga_buf[i+1], b_i, self.post_scale[i][0], self.post_scale[i][1])
            
    def predict ( self, inp, in_scale):
        row_padded, col_padded = self.init_databuf(inp.shape)
        self.loadInstr()
        
        padded_arr = np.zeros ( (row_padded, col_padded), dtype=inp.dtype, order='C')
        padded_arr[0:inp.shape[0], 0:inp.shape[1]] = inp
        
        print ("input shape", padded_arr.shape)
        np.copyto(self.fpga_buf[0], np.int16( padded_arr * in_scale ), casting='same_kind', where=True)
        gemx.sendMat(self.fpga_buf[0])
        gemx.execute()
        gemx.getMat (self.fpga_buf[-1])
        return self.fpga_buf[-1][:self.out_dim[0],:self.out_dim[1]]                
    