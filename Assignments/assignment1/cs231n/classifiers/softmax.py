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

    scores = np.dot(X, W)

    N, C = X.shape[0], W.shape[-1]
    for i in range(N):
      shifted_s = scores - np.max(scores, keepdims=True) # shift to avoid numeric instability in exp()
      e = np.exp(shifted_s)
      prob = e[i] / np.sum(e[i])
      loss += -np.log(prob[y[i]] + 1e-6)
      prob[y[i]] -= 1
      dW += np.outer(X[i], prob)
      

    loss = loss / N + reg * np.sum(W ** 2);
    dW = dW / N + 2 * reg * W


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
    
    N, C = X.shape[0], W.shape[-1] 
    scores = X.dot(W)
    shifted_s = scores - np.max(scores, axis=1, keepdims=True) # shift to avoid numerical instability in exp()
    e = np.exp(shifted_s)
    probs = e / np.sum(e, axis=1).reshape(-1, 1)
    loss += np.sum(-np.log(probs[range(N), y] + 1e-6)) / N + reg * np.sum(W ** 2)
    probs[range(N), y] -= 1
    dW += np.transpose(X) @ probs / N + 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
