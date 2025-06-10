from .Server.hor_comp import comp_part_server as cp_server
from .Server.comm import tcp_sockets_server as tcp_server

from .Client.comm import tcp_sockets_client as tcp_client
from .Client.hor_comp import comp_part_client as cp_client

def start_server(NumofClients, NumofRounds, model, ser_address):
    try:
        # Connections are established between the server and the clients
        # The number of connections is defined by the number of clients
        sockets = tcp_server.create_sockets(NumofClients, ser_address)

        # Once the connections are established, the model is exchanged
        # and the average model is computed
        avg_model = cp_server.compute_avg(sockets, NumofRounds, model)

        # Once the communication is done, the connections are closed
        tcp_server.close_sockets(sockets)

        return avg_model
    except Exception as e:
        print(f"Error during server execution: {e}")
        return None

def start_client(server_address, X_train, y_train, model, NumofRounds, epochs):
    try:
        # Connection is established between the server and the client
        socket = tcp_client.create_socket(server_address)

        # The client computes the local model on its local data
        avg_model = cp_client.comp_model(socket, X_train, y_train, model, NumofRounds, epochs)

        # The client will close the connection once the communication is done
        tcp_client.close_socket(socket)

        return avg_model
    except Exception as e:
        print(f"Error during client execution: {e}")
        return None