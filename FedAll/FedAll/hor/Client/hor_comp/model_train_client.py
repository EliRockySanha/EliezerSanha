def training(X_train, y_train, model, model_paras, epochs):
    import tensorflow as tf
    # To set the weights back into the model
    index = 0
    for layer in model.layers:
        if layer.weights:  # Check if the layer has weights
            weights = model_paras[index]
            biases = model_paras[index + 1]
            layer.set_weights([weights, biases])
            index += 2

    # Now fit the model with the provided training data
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=10, validation_split=0.1, verbose=0)


    # Extract the loss and accuracy for the last epoch
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]

    print(f'Training Loss: {final_loss}, Training Accuracy: {final_accuracy}')

    model_paras = []
    for layer in model.layers:
        weights, biases = layer.get_weights()
        model_paras.append(weights)
        model_paras.append(biases)

    return model_paras