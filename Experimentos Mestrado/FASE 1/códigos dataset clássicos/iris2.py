# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 02:12:01 2023

@author: rocky
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

class ParticleSwarmOptimizer:
    def __init__(self, input_shape, num_classes, n_particles=10, n_iter=100, learning_rate=0.01, inertia=0.9, c1=2.0, c2=2.0, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.swarm = self.initialize_swarm()
        self.best_positions = self.swarm.copy()
        self.best_scores = np.full(n_particles, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def initialize_swarm(self):
        return np.random.uniform(low=-1, high=1, size=(self.n_particles, self.get_flatten_params_count()))

    def get_flatten_params_count(self):
        model = self.build_model()
        return len(tf.keras.layers.Flatten()(model.get_weights()))

    def build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=self.input_shape),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def decode_particle(self, position):
        model = self.build_model()
        flat_params_count = len(tf.keras.layers.Flatten()(model.get_weights()))
        model.set_weights(tf.keras.utils.unflatten(np.reshape(position, flat_params_count), model.get_weights()))
        return model

    def evaluate_particle(self, position, X, y):
        model = self.decode_particle(position)
        indices = np.random.choice(len(X), size=self.batch_size, replace=False)
        X_batch, y_batch = X[indices], y[indices]
        y_one_hot = to_categorical(y_batch, num_classes=self.num_classes)
        score = model.evaluate(X_batch, y_one_hot, verbose=0)[1]
        return score

    def update_best_positions(self):
        scores = np.array([self.evaluate_particle(pos, self.X_train, self.y_train) for pos in self.swarm])
        improved_positions = np.where(scores < self.best_scores)[0]
        self.best_scores[improved_positions] = scores[improved_positions]
        self.best_positions[improved_positions] = self.swarm[improved_positions]

        best_global_index = np.argmin(scores)
        if scores[best_global_index] < self.global_best_score:
            self.global_best_score = scores[best_global_index]
            self.global_best_position = self.swarm[best_global_index]

    def update_swarm(self):
        r1, r2 = np.random.rand(self.n_particles, self.get_flatten_params_count()), np.random.rand(self.n_particles, self.get_flatten_params_count())

        inertia_term = self.inertia * self.swarm
        cognitive_term = self.c1 * r1 * (self.best_positions - self.swarm)
        social_term = self.c2 * r2 * (self.global_best_position - self.swarm)

        velocity = inertia_term + cognitive_term + social_term
        gradients = self.get_gradients(self.global_best_position, self.X_train, self.y_train)
        self.swarm += velocity - (self.learning_rate * gradients)

    def get_gradients(self, position, X, y):
        model = self.decode_particle(position)
        indices = np.random.choice(len(X), size=self.batch_size, replace=False)
        X_batch, y_batch = X[indices], y[indices]
        y_one_hot = to_categorical(y_batch, num_classes=self.num_classes)
        with tf.GradientTape() as tape:
            predictions = model(X_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_one_hot, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        flat_gradients = np.concatenate([tf.reshape(g, [-1]).numpy() for g in gradients])
        return flat_gradients

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        for i in range(self.n_iter):
            self.update_best_positions()
            self.update_swarm()

            if i % 10 == 0:
                print("Iteration {}, Best Score: {:.4f}".format(i, self.global_best_score))

        best_model = self.decode_particle(self.global_best_position)
        return best_model

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
y_one_hot = to_categorical(y, num_classes=3)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Create and train Particle Swarm Optimizer
optimizer = ParticleSwarmOptimizer(input_shape=(4,), num_classes=3, n_particles=10, n_iter=100, learning_rate=0.01)
best_model = optimizer.train(X_train, y_train)

# Evaluate on the test set
accuracy = best_model.evaluate(X_test, y_test, verbose=0)[1]
print("Accuracy on the test set:", accuracy)
