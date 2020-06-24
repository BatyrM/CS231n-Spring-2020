from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
         f = X[i].dot(W)
         f -= np.max(f) # avoiding numerical instability
         scores = np.exp(f) / np.sum(np.exp(f)) # scores was normalized
         loss += -np.log(scores[y[i]])
         for j in range(num_classes):
              dW[:, j] += scores[j] * X[i]
         dW[:, y[i]] -= X[i]

    dW /= num_train
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train, num_classes = X.shape[0], W.shape[1]
    scores = X.dot(W)
    scores -= np.max(scores, axis = 1, keepdims = True)
    scores = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)
    loss = np.sum(-np.log(scores[np.arange(num_train), y]))
    loss /= num_train
    loss += reg * np.sum(W * W)

    scores[np.arange(num_train), y] -= 1 # grad Wy_i of L_i = (exp(f[y_i]) / sum(exp(f[j])) - 1) * X_i
                                         # because of that we should decrement all scores[:, true classes in y] by one.
    dW = X.T.dot(scores)  # grad W_j of L_i = ( softmax[some j in num_classes] / softmax[all classes] * X_i)
                          # so we have dW = matrix product of scores and X as we can see from looped implementation above
    dW /= num_train
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
