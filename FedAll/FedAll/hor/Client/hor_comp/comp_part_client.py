from . import model_train_client as md
from ..comm import data_transfer_client as dt
from colorama import Fore, Style

def comp_model(socket, X_train, y_train, model, NumofRounds, epochs):
    try:
        for r in range(NumofRounds):
            # Receive the average model from the server
            model_paras = dt.receive_model_from_server(socket)

            print(Fore.BLUE + "Round: " + str(r + 1) + Style.RESET_ALL)

            # Start the training on the local data
            model_paras = md.training(X_train, y_train, model, model_paras, epochs)

            # Send the local model to the server
            dt.send_model_to_server(socket, model_paras)
        
        return model_paras

    except dt.DataTransferError as e:
        print("Data transfer error occurred:", e)
        # Close the socket to avoid any further communication
        socket.close()
        return None

    except md.TrainingError as e:
        print("Training error occurred:", e)
        # Close the socket to avoid any further communication
        socket.close()
        return None

    except Exception as e:
        print("An unexpected error occurred:", e)
        # Close the socket to avoid any further communication
        socket.close()
        return None