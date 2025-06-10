from ..comm import data_transfer_client as dt_client

def comp_model(client_socket, X_train, W1, NumofRounds):

    import numpy as np

    [trainval, feat] = X_train.shape

    # Computation at the client 1 side


    for r in range(NumofRounds):

        print("Round: "+str(r+1))

        Z1_1 = np.dot(W1, X_train.T)

        dt_client.send_Z_to_server(client_socket, Z1_1)

        received_dZ_matrix = dt_client.receive_dZ_from_server(client_socket)

        learning_rate = 0.01
        
        dW1 = (1/trainval)*np.dot(received_dZ_matrix, X_train)
        W1 = W1 - learning_rate*dW1