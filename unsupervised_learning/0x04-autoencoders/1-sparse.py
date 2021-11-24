#!/usr/bin/env python3
"""Module that contains """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Function that creates a sparse autoencoder.

    Args:
        input_dims (int): An integer containing the dimensions of the model
            input.
        hidden_layers (list[int]): A list containing the number of nodes for each
            hidden layer in the encoder, respectively. The hidden layers should
            be reversed for the decoder.
        latent_dims (int): An integer containing the dimensions of the latent
            space representation.
        lambtha (float): The regularization parameter used for L1
            regularization on the encoded output.

    Returns:
        encoder(keras.models.Model): The encoder model.
        decoder(keras.models.Model): The decoder model.
        auto(keras.models.Model): The sparse autoencoder model.
    """
    # Encoder
    enco_in = keras.Input(shape=(input_dims,))
    enco_hid = keras.layers.Dense(hidden_layers[0], activation='relu')(enco_in)
    if len(hidden_layers) > 1:
        for i in range(1, len(hidden_layers)):
            enco_hid = keras.layers.Dense(
                hidden_layers[i], activation='relu'
            )(enco_hid)
    enco_out = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
        )(enco_hid)

    encoder = keras.models.Model(inputs=enco_in, outputs=enco_out)

    # Decoder
    deco_in = keras.Input(shape=(latent_dims,))
    deco_hid = keras.layers.Dense(
        hidden_layers[-1], activation='relu'
    )(deco_in)
    if len(hidden_layers) > 1:
        for j in range(len(hidden_layers) - 2, -1, -1):
            deco_hid = keras.layers.Dense(
                hidden_layers[j], activation='relu'
            )(deco_hid)
    deco_out = keras.layers.Dense(input_dims, activation='sigmoid')(deco_hid)

    decoder = keras.models.Model(deco_in, deco_out)

    # Auto
    auto_in = encoder(enco_in)
    auto_out = decoder(auto_in)
    auto = keras.models.Model(inputs=enco_in, outputs=auto_out)
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    np.random.seed(0)
    tf.set_random_seed(0)
    encoder, decoder, auto = autoencoder(784, [128, 64], 32, 10e-6)
    auto.fit(x_train, x_train, epochs=100,batch_size=256, shuffle=True,
                    validation_data=(x_test, x_test))
    encoded = encoder.predict(x_test[:10])
    print(np.mean(encoded))
    reconstructed = decoder.predict(encoded)

    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        ax.axis('off')
        plt.imshow(x_test[i].reshape((28, 28)))
        ax = plt.subplot(2, 10, i + 11)
        ax.axis('off')
        plt.imshow(reconstructed[i].reshape((28, 28)))
    plt.show()
