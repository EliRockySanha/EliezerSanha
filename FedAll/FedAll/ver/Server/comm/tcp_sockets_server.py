def create_sockets(NumofClients, ser_address):
    import socket

    ip, port = ser_address.split(':')
    # Convert port to integer
    port = int(port)

    try:
        # Creating TCP Socket
        # socket.AF_INET: This constant represents the address family for IPv4
        # socket.SOCK_STREAM: This constant represents the socket type for a TCP 
        # (Transmission Control Protocol) socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # The SO_REUSEADDR option allows reusing local addresses
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # The socket is bind with the specified hostname and the port
        server_socket.bind((ip, port))

        # The socket is listening to the incoming connections request
        # with the specified hostname and the port
        server_socket.listen(1)

        # Accepting connections from the clients
        # It will wait for the specified number of requests
        # defined by the number of the clients
        sockets_client = []
        for _ in range(NumofClients):
            print("Waiting for incoming connections...")
            # The new connection is stored in a socket variable
            # The future communication will take place via this socket variable
            socket, client_address = server_socket.accept()

            print("Accepted connection from", client_address)

            # Putting all the sockets in one list
            sockets_client.append(socket)

        return sockets_client

    except socket.error as e:
        print("Error occurred while creating sockets:", e)
        # Close the sockets if any were opened before the error
        for sock in sockets_client:
            sock.close()
        return None

def close_sockets(sockets):
    try:
        for sock in sockets:
            sock.close()
    except Exception as e:
        print("Error occurred while closing sockets:", e)