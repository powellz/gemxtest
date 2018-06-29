import gemx
import numpy as np
import math

class GemxRT():
    def __init__(self, wgt,bias, in_dim, out_dim, wgt_scale, post_scale, xclbin_opt):
        self.min_m = 32*int(xclbin_opt["GEMX_gemmMBlocks"])
        self.min_k = 32*int(xclbin_opt["GEMX_gemmKBlocks"])
        self.min_n = 32*int(xclbin_opt["GEMX_gemmNBlocks"]) 
        self.w = [ np.int16(a*b) for a,b in zip(wgt, wgt_scale)]
        self.b = [ np.int32(a*b) for a,b in zip(bias, wgt_scale)]
        self.w = self.format_for_fpga( self.w, self.min_k, self.min_n)
        self.b = self.format_for_fpga( self.b, self.min_m, self.min_n)
        gemx.load_buf( self.w )
        gemx.load_buf( self.b )
        in_row, in_col = self.get_padded_shape(in_dim, self.min_m, self.min_k)
        self.fpga_buf = self.create_buf( self.w, [in_row,in_col])
        self.out_dim = out_dim 
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
    
    def create_buf ( self, q_wt, inp_shape):
        fpga_buf = []
        buf_shape = inp_shape
        fpga_buf.append ( gemx.create_fpga_buf( buf_shape, q_wt[0].dtype ) )
        for w in q_wt:
            buf_shape = ( buf_shape[0], w.shape[1] )
            fpga_buf.append ( gemx.create_fpga_buf( buf_shape, w.dtype ) )
        
        return fpga_buf             
    
    def loadInstr(self):
        gemx.clearInstrBuf()
        for i,(w_i,b_i) in enumerate( zip( self.w, self.b) ):
            gemx.addGEMMOp( self.fpga_buf[i], w_i , self.fpga_buf[i+1], b_i, self.post_scale[i][0], self.post_scale[i][1])
            
    def predict ( self, inp, in_scale):
        self.loadInstr()
        row_padded, col_padded = self.get_padded_shape( inp.shape, self.min_m, self.min_k)
        padded_arr = np.zeros ( (row_padded, col_padded), dtype=inp.dtype, order='C')
        padded_arr[0:inp.shape[0], 0:inp.shape[1]] = inp
        
        print ("input shape", padded_arr.shape)
        np.copyto(self.fpga_buf[0], np.int16( padded_arr * in_scale ), casting='same_kind', where=True)
        gemx.sendMat(self.fpga_buf[0])
        gemx.execute()
        gemx.getMat (self.fpga_buf[-1])
        return self.fpga_buf[-1][:self.out_dim[0],:self.out_dim[1]]                
   
class KerasRT(GemxRT):
    def __init__(self, keras_model, xclbin_opt, batch_sz, wgt_scale, post_scale):
        keras_w = keras_model.get_weights()[0::2]
        keras_b = keras_model.get_weights()[1::2]
        self.__init__(keras_w, keras_b, [batch_sz, keras_model.layers[0].input_shape[1]], 
                      [ batch_sz, keras_model.layers[-1].output_shape[1] ], 
                      batch_sz, wgt_scale, post_scale, xclbin_opt)
        self.kmodel = keras_model
        
    def loadInstr(self):
        gemx.clearInstrBuf()
        for i,l in enumerate(self.kmodel.layers):
            act = l.get_config()['activation']
            if act == 'relu':
                gemx.addFCNOp( self.fpga_buf[i], self.w[i], self.fpga_buf[i+1], self.b[i], self.post_scale[i][0], self.post_scale[i][1], 0, 0)
            else:
                gemx.addGEMMOp( self.fpga_buf[i], self.w[i], self.fpga_buf[i+1], self.b[i], self.post_scale[i][0], self.post_scale[i][1])
