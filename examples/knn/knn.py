import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from platform import dist
import gemx
from gemx_rt import GemxRT
import operator

class xKNN:
  def __init__(self, X, Y, in_shape, xclbin_opt):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = Y
    bias = np.zeros((in_shape[0], self.X_train.shape[0]), dtype=np.int32, order='C')
    self.gemxRT = GemxRT (xclbin_opt, X.T, bias, wgt_scale=[1], post_scale=[ [1, 0]])

  def predict_fpga(self, X, k=1):
    dists = self.compute_dist_fpga(X)
    return self.predict_labels(dists, k=k)

  def predict_cpu(self, X, k=1):
    """
    Predict labels for test data using this classifier.
    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    dists = self.compute_dist(X)
    return self.predict_labels(dists, k=k)

  def compute_dist_fpga(self, X):
    fpga_out = self.gemxRT.predict(X, 1)
    test_sum = np.sum(np.square(X), axis=1)
    train_sum = np.sum(np.square(self.X_train), axis=1)
    #print ("predict fpga", test_sum.shape, train_sum.shape, fpga_out.shape)
    dists = np.sqrt(-2 * fpga_out + test_sum.reshape(-1, 1) + train_sum)
    return dists  

  def compute_dist(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.
    """
    # Compute the l2 distance between all test points and all training
    # points without using any explicit loops, and store the result in
    # dists.                                                          
    # Output: sqrt((x-y)^2)
    # (x-y)^2 = x^2 + y^2 - 2xy
    test_sum = np.sum(np.square(X), axis=1)  # num_test x 1
    train_sum = np.sum(np.square(self.X_train), axis=1)
    inner_product = np.dot(X, self.X_train.T) 
    #print ("predict cpu", test_sum.shape, train_sum.shape, inner_product.shape)
    dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1, 1) + train_sum)
    return dists


  def getResponse(self,neighbors):
        classVotes = {}
        for x in neighbors:
            if x in classVotes:
                classVotes[x] += 1
            else:
                classVotes[x] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]
    
  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.
    """
    num_test = dists.shape[0]
    y_pred = []
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      
      # Use the distance matrix to find the k nearest neighbors of the ith    
      # training point, and use self.y_train to find the labels of these      
      # neighbors. Store these labels in closest_y.                           
      y_indices = np.argsort(dists[i, :], axis=0)
      closest_y = self.y_train[y_indices[:k]]
      y_pred.append( self.getResponse(closest_y ) )      

    return y_pred

args, xclbin_opt = gemx.processCommandLine()
gemx.createGEMMHandle(args, xclbin_opt)

# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
num_neighbor = 3
# loading training data
df = pd.read_csv('./iris.data', header=None, names=names)
print(df.head())

# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4])
y = np.array(df['class']) 

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Quantization of floating point input should be applied for better accuracy
#Cast and round input data to int16 for brevity
X_train_int = np.ascontiguousarray(np.rint(X_train), dtype=np.int16)
X_test_int = np.ascontiguousarray(np.rint(X_test), dtype=np.int16)

knn = KNeighborsClassifier(n_neighbors=num_neighbor)
# fitting the model
knn.fit(X_train_int, y_train)
# predict the response
pred_sklearn = knn.predict(X_test)

pred_cpu = []
pred_fpga = []
knnInst = xKNN(X_train_int, y_train , X_test.shape, xclbin_opt)
pred_cpu = knnInst.predict_cpu(X_test_int, num_neighbor)
pred_fpga = knnInst.predict_fpga(X_test_int, num_neighbor)

# evaluate accuracy
acc_sklearn = accuracy_score(y_test, pred_sklearn) * 100
print('\nsklearn classifier accuracy: %d%%' % acc_sklearn)

acc_cpu = accuracy_score(y_test, pred_cpu) * 100
print('\nCPU classifier accuracy: %d%%' % acc_cpu)

acc_fpga = accuracy_score(y_test, pred_fpga) * 100
print('\nFPGA classifier accuracy: %d%%' % acc_fpga)
