import numpy as np
import pandas

from FedAll import FedAll as fa

data_s = pandas.read_csv('ver_data/data_s.csv')
data_s = data_s.to_numpy()

X_server_train = data_s[:,0:2]
y_train = data_s[:,2]

L = 4 # Number of layers, including the input and output
nx = np.array([X_server_train.shape[1], 5, 5, 1])

W = []
b = []

np.random.seed(10)
for l in range(L-1):
    W.append(np.random.randn(nx[l+1],nx[l]))
    b.append(np.random.randn(nx[l+1],1))

NumofClients = 2
NumofRounds = 2000
ser_address = "127.0.0.1:5021"

fa.ver_server_start(NumofClients, ser_address, X_server_train, y_train, W, b, NumofRounds)