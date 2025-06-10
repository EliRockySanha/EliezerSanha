import numpy as np

def backward_prop(W, b, y_train, trainval, A):

    learning_rate = 0.01

    L = len(W)

    # Backward Propagation
    A_last = A[-1]
    dZ = [A[-1] - y_train]  # Gradient for the output layer
    dW = [np.dot(dZ[0], A[-2].T) / trainval]  # Weight gradient for the output layer
    db = [np.sum(dZ[0], axis=1, keepdims=True) / trainval]  # Bias gradient for the output layer

    for l in reversed(range(L - 1)):  # Iterate backwards over layers (excluding the input layer)
        dA = A[l+1] * (1 - A[l+1])  # Derivative of the activation function
        dZ.insert(0, np.dot(W[l+1].T, dZ[0]) * dA)  # Gradient for Z[l]
        dW.insert(0, np.dot(dZ[0], A[l].T) / trainval)  # Weight gradient for layer l
        db.insert(0, np.sum(dZ[0], axis=1, keepdims=True) / trainval)  # Bias gradient for layer l

    # Model updates
    for l in range(L):
        W[l] = W[l] - learning_rate * dW[l]
        b[l] = b[l] - learning_rate * db[l]
    
    return W, b, dZ[0]