import gemx
import numpy as np
import math

class KerasRT():
    def __init__(self, keras_model, batch_sz, wgt_scale, min_m, min_k, min_n):
        self.w = keras_model.get_weights()[0::2]
        self.b = keras_model.get_weights()[1::2]
        
        self.w = [ np.int16(a*b) for a,b in zip(self.w, wgt_scale)]
        self.b = [ np.int32(a*b) for a,b in zip(self.b, wgt_scale)]
    
        self.w = self.format_for_fpga( self.w, min_k, min_n)
        self.b = self.format_for_fpga ( self.b, min_m, min_n)
        gemx.load_buf( self.w )
        gemx.load_buf( self.b )
        in_row, in_col = self.get_padded_shape([batch_sz, keras_model.layers[0].input_shape[1]], min_m, min_k)
        self.fpga_buf = self.create_buf( self.w, [in_row,in_col])
        self.out_dim = ( batch_sz, keras_model.layers[-1].output_shape[1] )
        self.min_m = min_m
        self.min_k = min_k
        self.min_n = min_n

    def get_padded_shape ( self, shape, min_row, min_col):
        row_padded = int(math.ceil ( shape[0] / min_row ) * min_row ) 
        col_padded = int( math.ceil( shape[1] / min_col ) * min_col )
        return row_padded,col_padded

    def format_for_fpga ( self, np_list, min_row, min_col):
        padded_list = []
        for m in np_list:
            if m.ndim == 1:
                m = m.reshape(m.shape[0],1)
    
            row_padded, col_padded = self.get_padded_shape ( m.shape, min_row, min_col)
            padded_arr = np.zeros ( (row_padded, col_padded), dtype=m.dtype, order='C')
            padded_arr[0:m.shape[0], 0:m.shape[1]] = m
    #        print ("padded shape", padded_arr.shape)  
    #        print (padded_arr)            
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
    
    def predict ( self, inp, in_scale, post_scale):
        activations = ['relu', 'relu', 'none']
        row_padded, col_padded = self.get_padded_shape( inp.shape, self.min_m, self.min_k)
        padded_arr = np.zeros ( (row_padded, col_padded), dtype=inp.dtype, order='C')
        padded_arr[0:inp.shape[0], 0:inp.shape[1]] = inp
        
        print ("input shape", padded_arr.shape)
        np.copyto(self.fpga_buf[0], np.int16( padded_arr * in_scale ), casting='same_kind', where=True)
        gemx.sendMat(self.fpga_buf[0])
        for i,iw in enumerate(self.w):
            print ("w" ,i, "shape", iw.shape)
            if activations[i] == 'relu':
                gemx.addFCNOp( self.fpga_buf[i], iw, self.fpga_buf[i+1], self.b[i], post_scale[i][0], post_scale[i][1], 0, 0)
            else:
                gemx.addGEMMOp( self.fpga_buf[i], iw, self.fpga_buf[i+1], self.b[i], post_scale[i][0], post_scale[i][1])
                 
        gemx.execute()
        gemx.getMat (self.fpga_buf[-1])
        return self.fpga_buf[-1][:self.out_dim[0],:self.out_dim[1]]    
