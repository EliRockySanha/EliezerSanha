from FedAll import FedAll as fa
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_load():
    data = pd.read_csv('data/data_3.csv')

    X = data[['feature A', 'feature B', 'feature C', 'feature D']]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train

def model_define():
    model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(4,)),
            tf.keras.layers.Dense(5, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
            tf.keras.layers.Dense(5, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
            tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))
        ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

X_train, y_train = data_load()
model = model_define()

server_address = "127.0.0.1:5007"
NumofRounds = 20
epochs = 2

final_model = fa.hor_client_start(server_address, X_train, y_train, model, NumofRounds, epochs)