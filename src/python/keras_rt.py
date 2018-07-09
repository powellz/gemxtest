import gemx
import numpy as np
import math
from gemx_rt import GemxRT
        
class KerasRT(GemxRT):
    def __init__(self, keras_model, xclbin_opt, wgt_scale, post_scale):
        keras_w = keras_model.get_weights()[0::2]
        keras_b = keras_model.get_weights()[1::2]
        super().__init__(keras_w, keras_b, wgt_scale, post_scale, xclbin_opt)
        self.kmodel = keras_model
       
    def loadInstr(self):
        gemx.clearInstrBuf()
        for i,l in enumerate(self.kmodel.layers):
            act = l.get_config()['activation']
            if act == 'relu':
                gemx.addFCNOp( self.fpga_buf[i], self.w[i], self.fpga_buf[i+1], self.b[i], self.post_scale[i][0], self.post_scale[i][1], 0, 0)
            else:
                gemx.addGEMMOp( self.fpga_buf[i], self.w[i], self.fpga_buf[i+1], self.b[i], self.post_scale[i][0], self.post_scale[i][1])
