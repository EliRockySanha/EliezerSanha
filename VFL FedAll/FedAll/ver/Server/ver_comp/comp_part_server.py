from ..comm import data_transfer_server as dt_server
from . import forward_prop_model as md_for
from . import backward_prop_model as md_back

import numpy as np

def model_train(NumofClients, X_train, y_train, W, b, NumofRounds, sockets_client):
    [trainval, feat] = X_train.shape

    L = len(W)

    learning_rate = 0.01

    # Training
    J = np.zeros((NumofRounds,1))
    train_accuracy = np.zeros((NumofRounds,1))

    for r in range(NumofRounds):
            
            Z1_c = dt_server.receive_all_Zs_from_clients(sockets_client)

            Z, A = md_for.forward_prop(W, b, X_train, Z1_c, NumofClients)

            # Loss function
            J[r] = (-1/trainval)*np.sum(y_train.T*np.log(A[-1]) + (1-y_train.T)*np.log(1-A[-1]))
            
            # Accuracy
            pred_train = A[-1] > 0.5
            train_accuracy[r] = 1 - np.sum(abs(pred_train - y_train))/trainval 

            print(f"\033[94mRound: {r+1}\033[0m | Loss: {float(J[r]):.4f} | Accuracy: {float(train_accuracy[r]):.4f}")

            W, b, dZ1 = md_back.backward_prop(W, b, y_train, trainval, A)

            dt_server.send_dZs_to_all_clients(dZ1, sockets_client)
            