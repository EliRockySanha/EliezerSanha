from FedAll import FedAll as fa
import tensorflow as tf

def int_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),
        tf.keras.layers.Dense(5, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
        tf.keras.layers.Dense(5, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))
    ])

    model_paras = []
    for layer in model.layers:
        weights, biases = layer.get_weights()
        model_paras.append(weights)
        model_paras.append(biases)

    return model_paras

initial_model_paras = int_model()

# The server can set the values for the following hyper parameters
NumofClients = 3
NumofRounds = 20

server_add = "127.0.0.1:5007"

final_model = fa.hor_server_start(NumofClients, NumofRounds, initial_model_paras, server_add) 