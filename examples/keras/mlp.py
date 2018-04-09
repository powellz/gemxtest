
from __future__ import print_function
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.callbacks import Callback, ModelCheckpoint
import argparse
import math

def train(train_fd, predictors, train_data, num_classes):
    
    # We will use a multi-layer perceptron classification model for random search.
    # Create model
    #estimator = KerasClassifier(build_fn=create_keras_model(len(predictors), len(train[target].unique())), epochs=200, batch_size=5, verbose=0)
    model = create_keras_model(len(predictors), num_classes )
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    modelcheckpoint_callback = ModelCheckpoint("./best_model.h5", monitor='val_loss',mode='min', save_best_only=True, save_weights_only=True)
    
    model.fit(train_fd[predictors], train_data, epochs=200, batch_size=50, callbacks=[modelcheckpoint_callback], validation_split=0.20, shuffle=True)

def predict_hwemu ( weights, input_dim, test_data, num_classes, use_fpga = False ):
    model = create_keras_model(input_dim, num_classes )
    model.load_weights(weights)
    #return compute_standalone( test_data.values, model.get_weights())    
    return compute_standalone_hwemu( test_data.values, model.get_weights())

def predict_cpu ( weights, input_dim, test_data, num_classes ):
    model = create_keras_model(input_dim, num_classes )
    model.load_weights(weights)
    predictions = model.predict(test_data)
    
#     layer_name = 'd1'
#     intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(layer_name).output)
#     intermediate_output = intermediate_layer_model.predict(test_data)
    #return intermediate_output
    return predictions

def format_for_fpga ( np_list, min_row, min_col):
    padded_list = []
    for m in np_list:
        row_padded = int(math.pow(2, int( math.ceil (math.log(m.shape[0], 2)) )))
        row_padded = max (row_padded, min_row)
        
        if m.ndim == 1:
            m = m.reshape(m.shape[0],1)
            
        col_dim = min_col if len(m.shape) == 1 else m.shape[1]
        col_padded = int(math.pow(2, int( math.ceil( math.log(col_dim, 2) ) )))
        col_padded = max (col_padded, min_col)
        padded_arr = np.zeros ( (row_padded, col_padded), dtype=m.dtype, order='C')
        padded_arr[0:m.shape[0], 0:col_dim] = m
          
#        print ("padded shape", padded_arr.shape)  
#        print (padded_arr)            
        padded_list.append(padded_arr)

    return padded_list
    
def predict_fpga( args, test_data, num_classes ):
    import gemx
    model = create_keras_model(test_data.values.shape[1], num_classes )
    model.load_weights(args.model)
    out_dim = ( test_data.values.shape[0], model.layers[-1].output_shape[1] )
    weights = model.get_weights()[0::2]
    bias = model.get_weights()[1::2]
        
    #load xclbin 
    gemx.createFCNHandle( args.xclbin, args.kernelName, args.gemxlib , args.device )
    
    #Quantization parameters to bring fp32 ranges to fit into int16; parameters are derived offline
    wgt_scale = [155275.3311, 121798.1115, 71553.71463]
    post_scale = [ [2,18], [2,18] , [2,17] ]    
    in_scale = 31.13053392
    
    #quantize weights and bias
    w_int16 = [ np.int16(a*b) for a,b in zip(weights, wgt_scale)]
    b_int32 = [ np.int32(a*b) for a,b in zip(bias, wgt_scale)]
    
    #pad matrices to minimum dimensions as required by the FPGA kernel
    padded_w = format_for_fpga( w_int16, 128, 128)
    padded_b = format_for_fpga ( b_int32, 128, 128)
    padded_in = format_for_fpga ( [test_data.values], 128, 128)
    
    #create intermediate buffers on FPGA    
    fpga_buf = gemx.create_buf( padded_w, padded_in[0].shape)
    
    #load weights 
    gemx.load_buf( padded_w )
    gemx.load_buf( padded_b )
    
    #activation types after each GEMM operation 
    a = ['relu', 'relu', 'none']
    
    #run inference on FPGA
    result = gemx.predict( padded_w, padded_b, a, fpga_buf, padded_in[0], in_scale, post_scale, out_dim )
    
    #run softmax on CPU
    for i in range(result.shape[0]):
        result[i,:] = softmax(result[i,:])
        
    return result
 
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_dense(weight, bias, inp, scalein=1, post_scale=1):
    scaledin = inp*scalein
    inp16 = np.int16(scaledin)#input from previous stage to 16bits
    m64 = np.matmul(np.int64(inp16), np.int64(weight))#intermediate accumulation to 64 bits
    
    output64 = m64
    if bias is not None:
        bias64 = np.int64(bias)#bias to 64 bits
        output64 = m64 + bias64
        
    o64d = output64/(2**post_scale[1])
    o64m = o64d*post_scale[0]
    output = np.int16(o64m)#scale down for 16 bits
    return output
    
def compute_standalone_hwemu( inp, wb ):
    in_scale = [31.13053392, 1, 1]
    wgt_scale = [155275.3311, 121798.1115, 71553.71463]
    post_scale = [ 43.88706261, 39.40462421,41.47343054 ] 
    #post_scale_hw = [ [44,4833804], [39.4,5345361] , [1, 2819547] ]    
    post_scale_hw = [ [2,18], [2,18] , [2, 17] ]    

    weights = wb[0::2]
    bias = wb[1::2]

    #quantization
    w_int16 = [ np.int16(a*b) for a,b in zip(weights, wgt_scale)]
    b_int32 = [ np.int32(a*b) for a,b in zip(bias, wgt_scale)]
        
    o1 = compute_dense ( w_int16[0], b_int32[0], inp, in_scale[0], post_scale_hw[0])
    print ("o1 range (", np.min(o1), ",", np.max(o1), ")")
    o1[o1 < 0] = 0
    o2 = compute_dense ( w_int16[1], b_int32[1], o1, in_scale[1], post_scale_hw[1])
    print ("o2 range (", np.min(o2), ",", np.max(o2), ")")    
    o2[o2 < 0] = 0

    o3 = compute_dense ( w_int16[2], b_int32[2], o2, in_scale[2], post_scale_hw[2])

    print ("o3 range (", np.min(o3), ",", np.max(o3), ")")
    #softmax
    for i in range(o3.shape[0]):
        o3[i,:] = softmax(o3[i,:])
    return o3

def compute_standalone( inp, wb ):
    print ("inp (", np.min(inp), ",", np.max(inp))
    for i,w in enumerate(wb):
        print ( "w", i, ": ", np.min(w), ", ", np.max(w))
        
    o1 = np.matmul ( inp, wb[0])
    o1 = o1 + wb[1]
    print ("o1 (", np.min(o1), ",", np.max(o1))
    o1[o1 < 0] = 0
    o2 = np.matmul ( o1, wb[2])
    o2 = o2 + wb[3]
    print ("o2 (", np.min(o2), ",", np.max(o2))    
    o2[o2 < 0] = 0
    o3 = np.matmul ( o2, wb[4])
    print ("o3 (", np.min(o3), ",", np.max(o3))    
    o3 = o3 + wb[5]
    #softmax
    for i in range(o3.shape[0]):
        o3[i,:] = softmax(o3[i,:])
    return o3
 
def create_keras_model(in_dims, num_classes):
    '''
    Generate a simple Keras model.
    '''  
    model = Sequential()
    model.add(Dense(100, input_dim=in_dims, activation='relu', name='d1'))
    model.add(Dense(25, activation='relu', name='d2'))
    model.add(Dense(num_classes, activation='softmax', name='d3'))
    #model.add(Dense(num_classes, name='d3'))
    model.summary()
    return model

if  __name__ == '__main__':
    np.random.seed(27)
    parser = argparse.ArgumentParser(description='GEMX')
    parser.add_argument('--data', required = True, help='inference data file')
    parser.add_argument('--model', required = True, help='model')
    parser.add_argument('--device', required = True, choices=['cpu', 'ku115','kcu1500','vu9p', 'vcu1525', 'vu9pf1'], help='supported FPGA devices')    
    parser.add_argument('--xclbin', help='file path to FPGA bitstream')
    parser.add_argument('--gemxlib', help='file path to GEMX host code shared library')
    parser.add_argument('-k', '--kernelName', default="gemxKernel_0", help='FPGA kernel name')  
    args = parser.parse_args()
    
    if args.device is not 'cpu':
        if args.xclbin is None or args.gemxlib is None:
             parser.error("FPGA execution requires --xclbin and --gemxlib")
             
    train_fd = pd.read_csv(args.data) # Load training data.
    IDcol = 'Run' # A column used to identified the run for data collection; not an independent variable.
    target = 'Class' # The column name for our dependent variable.
    predictors = [x for x in train_fd.columns if x not in [target, IDcol]] # Define column names to use as independent variables.
    
    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(train_fd[target])
    encoded_Y = encoder.transform(train_fd[target])
    # Convert integers to dummy variables (i.e. one hot encoded)
    train_y = np_utils.to_categorical(encoded_Y)
    
    hwemu_out = predict_hwemu( args.model, len(predictors), train_fd[predictors], len(train_fd[target].unique()) )
    fpga_out = predict_fpga( args, train_fd[predictors], len(train_fd[target].unique()))
    #cpu_out = predict_cpu( args.model, len(predictors), train_fd[predictors], len(train_fd[target].unique()) )
      
    if np.array_equal (hwemu_out, fpga_out):
        print ("SUCCESS!!!")
    else:
        np.savetxt("out.np", fpga_out, fmt="%f")
        np.savetxt("out_golden.np", hwemu_out, fmt="%f")
        np.savetxt("diff.np", hwemu_out - fpga_out, fmt="%f")