import gemx

class KerasRT():
    def __init__(self, keras_model, wgt_scale, post_scale, min_m, min_k, min_n):
        self.w = keras_model.get_weights()[0::2]
        self.b = keras_model.get_weights()[1::2]
        
        self.w = [ np.int16(a*b) for a,b in zip(self.w, wgt_scale)]
        self.b = [ np.int32(a*b) for a,b in zip(self.b, wgt_scale)]
    
        padded_w = format_for_fpga( self.w, min_k, min_n)
        padded_b = format_for_fpga ( self.b, min_m, min_n)
        padded_in = format_for_fpga ( [test_data.values], min_m, min_k)

    def format_for_fpga ( np_list, min_row, min_col):
        padded_list = []
        for m in np_list:
            row_padded = int(math.ceil ( m.shape[0] / min_row ) * min_row ) 
            if m.ndim == 1:
                m = m.reshape(m.shape[0],1)
                
            col_dim = min_col if len(m.shape) == 1 else m.shape[1]
            col_padded = int( math.ceil( col_dim / min_col ) * min_col )
            padded_arr = np.zeros ( (row_padded, col_padded), dtype=m.dtype, order='C')
            padded_arr[0:m.shape[0], 0:col_dim] = m
    #        print ("padded shape", padded_arr.shape)  
    #        print (padded_arr)            
            padded_list.append(padded_arr)
    
        return padded_list        
    
    def create_buf ( q_wt, inp_shape):
        fpga_buf = []
        buf_shape = inp_shape
        fpga_buf.append ( create_fpga_buf( buf_shape, q_wt[0].dtype ) )
        for w in q_wt:
            buf_shape = ( buf_shape[0], w.shape[1] )
            fpga_buf.append ( create_fpga_buf( buf_shape, w.dtype ) )
        
        return fpga_buf
    
    def predict ( w, b, activations, fpga_buf, inp, in_scale, post_scale, out_dim):
        np.copyto(fpga_buf[0], np.int16( inp * in_scale ), casting='same_kind', where=True)
        gemx.sendMat(fpga_buf[0])
        for i,iw in enumerate(w):
            if activations[i] == 'relu':
                gemx.addFCNOp( fpga_buf[i], iw, fpga_buf[i+1], b[i], post_scale[i][0], post_scale[i][1], 0, 0)
            else:
                gemx.addGEMMOp( fpga_buf[i], iw, fpga_buf[i+1], b[i], post_scale[i][0], post_scale[i][1])
                 
        gemx.execute()
        gemx.getMat (fpga_buf[-1])
        return fpga_buf[-1][:out_dim[0],:out_dim[1]]    