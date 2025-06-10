import pickle

def receive_dZ_from_server(client_socket):
    dZ_length = int.from_bytes(client_socket.recv(4), byteorder='big')
    # Receive the actual data
    received_dZ = b''
    while len(received_dZ) < dZ_length:
            part = client_socket.recv(4096)
            received_dZ += part

    received_dZ_matrix = pickle.loads(received_dZ)

    return received_dZ_matrix

def send_Z_to_server(client_socket, Z1_1):
    Z1_1_bytes = pickle.dumps(Z1_1)
    # Send length of data first
    client_socket.sendall(len(Z1_1_bytes).to_bytes(4, byteorder='big'))
    # Send the actual data
    client_socket.sendall(Z1_1_bytes)