import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import math

from pso_adam1 import pso
import utils

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados para média zero e variância unitária
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = tf.cast(X_train, dtype=tf.float32)
X_test = tf.cast(X_test, dtype=tf.float32)

# Codificar one-hot nos rótulos
y_train = tf.one_hot(y_train, depth=3)  # 3 classes no conjunto de dados Iris
y_test = tf.one_hot(y_test, depth=3)

def multilayer_perceptron(weights, biases, X, x_min=0, x_max=1):
    """It runs the multilayer perceptron neural network with ReLU activation for hidden layers and Softmax for the output layer.

    Args:
        weights (List[tf.Tensor]): The weights of the neural net.
        biases (List[tf.Tensor]): The biases of the neural net.
        X (tf.Tensor): The input values.
        x_min (int, optional): The floor value for the normalization. Defaults to -1.
        x_max (int, optional): The roof value for the normalization. Defaults to 1.

    Returns:
        tf.Tensor: The prediction `Y`.
    """
    num_layers = len(weights) + 1
    H = 2.0 * (X - x_min) / (x_max - x_min) - 1.0

    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.relu(tf.add(tf.matmul(H, W), b))  # Use ReLU activation

    W = weights[-1]
    b = biases[-1]
    Y = tf.nn.softmax(tf.add(tf.matmul(H, W), b))  # Use Softmax for the output layer
    return Y

def get_loss(X, y):
    def _loss(w, b):
        with tf.GradientTape() as tape:
            tape.watch(w)
            tape.watch(b)
            pred = multilayer_perceptron(w, b, X)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, pred))
        trainable_variables = w + b
        grads = tape.gradient(loss, trainable_variables)
        return loss, grads

    return _loss

# Parâmetros
layers = [4] + 1*[20] + [3]
pop_size = 25
n_iter = 1000
sample_size = 512
noise = 0.0

opt = pso(
    get_loss(X_train,y_train),
    layers,
    n_iter,
    pop_size,
    0.999,
    8e-1,
    5e-1,
    initialization_method="xavier",
    verbose=True,
    gd_alpha=0.01,
)

inicio = time.time()
opt.train()
fim = time.time()
print("\nTempo de treinamento: ", fim - inicio)

nn_w, nn_b = opt.get_best()
pesosbiases = utils.encode(nn_w, nn_b)
#print(pesosbiases)


def get_accuracy(y, y_pred):
    """
    Calculates accuracy.
    :param y: tf.Tensor
        Correct labels in one-hot format.
    :param y_pred: tf.Tensor
        Predicted labels of shape (N, C) where C is the number of classes.
    :return: tf.Tensor
        Accuracy.
    """
    y_true = tf.argmax(y, axis=1)  # Convert one-hot labels to class indices

    y_pred = tf.argmax(y_pred, axis=1)  # Convert predicted labels to class indices

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))
    return accuracy

y_pred = multilayer_perceptron(nn_w, nn_b, X_test)

Acc = get_accuracy(y_test, y_pred)

print("Acurácia: %.3f"% Acc)