# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 00:33:20 2023

@author: Eliezer
"""
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pso_adam1 import pso
import utils
import time

# Carregar o conjunto de dados Sonar
sonar = fetch_openml(name="sonar", version=1)
X = sonar.data
y = sonar.target

# Converter rótulos para 0 (mina) e 1 (rocha)
y = (y == 'R').astype(int)

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados para média zero e variância unitária
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = tf.cast(X_train, dtype=tf.float32)
X_test = tf.cast(X_test, dtype=tf.float32)

# Codificar one-hot nos rótulos
y_train = tf.one_hot(y_train, depth=2)  # 2 classes no conjunto de dados Sonar
y_test = tf.one_hot(y_test, depth=2)

def multilayer_perceptron(weights, biases, X, x_min=0, x_max=1):
    """Executa a rede neural de perceptrons multicamadas com ativação ReLU para camadas ocultas e Softmax para a camada de saída.

    Args:
        weights (List[tf.Tensor]): Os pesos da rede neural.
        biases (List[tf.Tensor]): Os viéses da rede neural.
        X (tf.Tensor): Os valores de entrada.
        x_min (int, opcional): O valor mínimo para normalização. Padrão é -1.
        x_max (int, opcional): O valor máximo para normalização. Padrão é 1.

    Returns:
        tf.Tensor: A predição `Y`.
    """
    num_layers = len(weights) + 1
    H = 2.0 * (X - x_min) / (x_max - x_min) - 1.0

    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.relu(tf.add(tf.matmul(H, W), b))  # Use ativação ReLU

    W = weights[-1]
    b = biases[-1]
    Y = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))  # Use Sigmoid para a camada de saída (binária)
    return Y

def get_loss(X, y):
    def _loss(w, b):
        with tf.GradientTape() as tape:
            tape.watch(w)
            tape.watch(b)
            pred = multilayer_perceptron(w, b, X)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, pred))  # Use binary crossentropy para classificação binária
        trainable_variables = w + b
        grads = tape.gradient(loss, trainable_variables)
        return loss, grads

    return _loss


# Parâmetros
layers = [X_train.shape[1]] + 1*[20] + [2]  # Ajustar a entrada e a saída para o conjunto de dados Sonar
pop_size = 25
n_iter = 1000
sample_size = 512
noise = 0.0

opt = pso(
    get_loss(X_train, y_train),
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

def get_accuracy(y, y_pred):
    y_true = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))
    return accuracy

y_pred = multilayer_perceptron(nn_w, nn_b, X_test)
Acc = get_accuracy(y_test, y_pred)

print("Acurácia: %.3f"% Acc)

