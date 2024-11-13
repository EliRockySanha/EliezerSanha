
import time
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o conjunto de dados de câncer de mama
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados para média zero e variância unitária
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Codificar one-hot nos rótulos
y_train = tf.one_hot(y_train, depth=2)  # 2 classes no conjunto de dados de câncer de mama
y_test = tf.one_hot(y_test, depth=2)

# Parameters
input_dim = X_train.shape[1]  # Dimensão de entrada
output_dim = 2  # Número de classes
hidden_layers = [20]  # Camadas ocultas
learning_rate = 0.01
epochs = 1000
batch_size = 512

# Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
for units in hidden_layers:
    model.add(tf.keras.layers.Dense(units, activation='relu'))
model.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
start_time = time.time()
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
end_time = time.time()
print("\nTraining Time: ", end_time - start_time)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy:.3f}')
