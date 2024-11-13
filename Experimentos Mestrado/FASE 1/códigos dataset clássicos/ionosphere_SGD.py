import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Carregar o conjunto de dados Ionosphere
ionosphere_data = pd.read_csv('Ionosphere.csv')

# Separar características (X) e rótulos (y)
X = ionosphere_data.drop('Class', axis=1).values
y = ionosphere_data['Class'].values

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados para média zero e variância unitária
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Codificar one-hot nos rótulos
y_train = tf.one_hot(y_train, depth=len(set(y)))  # Número de classes no conjunto de dados Ionosphere
y_test = tf.one_hot(y_test, depth=len(set(y)))

def multilayer_perceptron(weights, biases, X, x_min=0, x_max=1):
    """Executa a rede neural de perceptrons multicamadas com ativação ReLU para camadas ocultas e Sigmoid para a camada de saída.

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
layers = [X_train.shape[1]] + 1*[20] + [len(set(y))]  # Ajustar a entrada e a saída para o conjunto de dados Ionosphere
learning_rate = 0.01
epochs = 1000
batch_size = 512

# Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
for units in layers[1:]:
    model.add(tf.keras.layers.Dense(units, activation='relu'))
model.add(tf.keras.layers.Dense(len(set(y)), activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
start_time = time.time()
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
end_time = time.time()
print("\nTraining Time: ", end_time - start_time)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy:.3f}')
