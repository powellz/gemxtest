import numpy as np
import gemx
from gemxRT import GemxRT

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """
  def __init__(self, xclbin_opt, in_dim, wgt_scale, post_scale, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y
    bias = np.zeroes( in_dim[0], self.X_train.shape[1])
    self.gemxRT= GemxRT( X_train, bias, in_dim, [in_dim[0], self.X_train.shape[1]], wgt_scale, post_scale,
                         xclbin_opt )

  def predict(self, X, k=1):
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
    fpga_out = test.test_basic(0, X, self.X_train.T)
    
    test_sum = np.sum(np.square(X), axis=1) # num_test x 1
    train_sum = np.sum(np.square(self.X_train), axis=1) # num_train x 1
    inner_product = np.dot(X, self.X_train.T) # num_test x num_train
    
    inner_product = self.gemxRT.predict ()
    dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1, 1) + train_sum) # broadcast
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return dists

  def compute_dist(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # Output: sqrt((x-y)^2)
    # (x-y)^2 = x^2 + y^2 - 2xy
    test_sum = np.sum(np.square(X), axis=1) # num_test x 1
    train_sum = np.sum(np.square(self.X_train), axis=1) # num_train x 1
    inner_product = np.dot(X, self.X_train.T) # num_test x num_train
    dists = np.sqrt(-2 * inner_product + test_sum.reshape(-1, 1) + train_sum) # broadcast
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      y_indicies = np.argsort(dists[i, :], axis = 0)
      closest_y = self.y_train[y_indicies[:k]]
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = np.argmax(np.bincount(closest_y))
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

