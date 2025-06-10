from .Client.ver_comp import comp_part_client as cp_client
from .Server.ver_comp import comp_part_server as cp_server

from .Server.comm import tcp_sockets_server as tcp_server
from .Client.comm import tcp_sockets_client as tcp_client

def start_server(NumofClients, server_address, X_train, y_train, W, b, NumofRounds):

    sockets_client = tcp_server.create_sockets(NumofClients, server_address)

    cp_server.model_train(NumofClients, X_train, y_train, W, b, NumofRounds, sockets_client)

    tcp_server.close_sockets(sockets_client)

def start_client(server_address, X_train, W1, NumofRounds):

    client_socket = tcp_client.create_socket(server_address)

    cp_client.comp_model(client_socket, X_train, W1, NumofRounds)

    tcp_client.close_socket(client_socket)   