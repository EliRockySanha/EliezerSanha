import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
import time

# Carregar o conjunto de dados Wine
wine = load_wine()
X = wine.data
y = wine.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados para média zero e variância unitária
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = tf.cast(X_train, dtype=tf.float32)
X_test = tf.cast(X_test, dtype=tf.float32)

# Codificar one-hot nos rótulos
y_train = tf.one_hot(y_train, depth=3)  # 3 classes no conjunto de dados Wine
y_test = tf.one_hot(y_test, depth=3)

# Definir a arquitetura da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compilar o modelo com RMSprop
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
inicio = time.time()
model.fit(X_train, y_train, epochs=1000, batch_size=512, verbose=1)
fim = time.time()
print("\nTempo de treinamento: ", fim - inicio)

# Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Acurácia: {accuracy:.3f}')

