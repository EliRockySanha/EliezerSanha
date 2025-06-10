import numpy as np

# Non linear sigmoid activation function
def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig

def forward_prop(W, b, X_train, Z1_c, NumofClients):

    L = len(W)
    
    Z1_s = np.dot(W[0], X_train.T) + b[0]
        
    Z1_t = Z1_s

    for client in range(NumofClients):
        Z1_t = Z1_t + Z1_c[client]

    Z = []
    A = []
    A.append(X_train.T)

    Z.append(Z1_t)
    A.append(sigmoid(Z[0]))

    for l in range(L-1):
        Z.append(np.dot(W[l+1], A[l+1]) + b[l+1])
        A.append(sigmoid(Z[l+1]))
    
    return Z, A