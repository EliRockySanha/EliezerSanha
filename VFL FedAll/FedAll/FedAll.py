from .hor import hor
from .ver import ver

def hor_server_start(NumofClients, NumofRounds, model, ser_address):
    return hor.start_server(NumofClients, NumofRounds, model, ser_address)

def hor_client_start(server_address, X_train, y_train, model, NumofRounds, epochs):
    return hor.start_client(server_address, X_train, y_train, model, NumofRounds, epochs)

def ver_client_start(server_address, X_train, W, NumofRounds):
    return ver.start_client(server_address, X_train, W, NumofRounds)

def ver_server_start(NumofClients, server_address, X_train, y_train, W, b, NumofRounds):
    return ver.start_server(NumofClients, server_address, X_train, y_train, W, b, NumofRounds)