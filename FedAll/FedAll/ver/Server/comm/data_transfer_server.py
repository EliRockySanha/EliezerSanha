import pickle

def send_dZs_to_all_clients(dZ, sockets_client):
    for sock in sockets_client:
        data_to_send = pickle.dumps(dZ)
        # Send length of data first
        sock.sendall(len(data_to_send).to_bytes(4, byteorder='big'))
        # Send the actual data
        sock.sendall(data_to_send)

def receive_all_Zs_from_clients(sockets_client):
    Z1_c = []
    for _, sock in enumerate(sockets_client):   
    
        data_length = int.from_bytes(sock.recv(4), byteorder='big')
        # Receive the actual data
        received_data = b''
        while len(received_data) < data_length:
                part = sock.recv(4096)
                received_data += part

        Z1_client = pickle.loads(received_data)
        Z1_c.append(Z1_client)
    
    return Z1_c