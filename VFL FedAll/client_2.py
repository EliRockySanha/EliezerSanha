import numpy as np
import pandas

from FedAll import FedAll as fa

data = pandas.read_csv('ver_data/data_2.csv')
data = data.to_numpy()
X_train = data

W1_1 = np.random.randn(5, X_train.shape[1])

NumofRounds = 2000

server_address = "127.0.0.1:5021"

fa.ver_client_start(server_address, X_train, W1_1, NumofRounds)