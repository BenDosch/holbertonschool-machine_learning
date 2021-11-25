#!/usr/bin/env python3
"""Module that contains a function that creates a variational autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates a variational autoencoder.

    Args:
        input_dims (int): An integer containing the dimensions of the model
            input.
        hidden_layers (list[int]): A list containing the number of nodes for
            each hidden layer in the encoder, respectively. The hidden layers
            should be reversed for the decoder.
        latent_dims (int): An integer containing the dimensions of the latent
            space representation.

    Returns:
        encoder(keras.models.Model): The encoder model, which should output the
            latent representation, the mean, and the log variance,
            respectively.
        decoder(keras.models.Model): The decoder model.
        auto(keras.models.Model): The full autoencoder model.
    """
    # Encoder
    enco_in = keras.Input(shape=(input_dims,))
    enco_hid = keras.layers.Dense(hidden_layers[0], activation='relu')(enco_in)
    if len(hidden_layers) > 1:
        for i in range(1, len(hidden_layers)):
            enco_hid = keras.layers.Dense(
                hidden_layers[i], activation='relu'
            )(enco_hid)

    μ = keras.layers.Dense(latent_dims, activation=None)(enco_hid)
    log_σ = keras.layers.Dense(latent_dims, activation=None)(enco_hid)

    def sampler(args):
        """sampler function for latent layer calculation"""
        μ, log_σ = args
        ε = keras.backend.random_normal(keras.backend.shape(μ))
        return μ + keras.backend.exp(0.5 * log_σ) * ε  # latent_vector

    z = keras.layers.Lambda(sampler)([μ, log_σ])

    encoder = keras.models.Model(inputs=enco_in, outputs=(z, μ, log_σ))

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

    def KL_loss(inputs, outputs):
        """cost function"""
        loss = keras.backend.binary_crossentropy(inputs, outputs)
        loss = keras.backend.sum(loss, axis=1)
        KL_divergence = (
            -0.5 * keras.backend.sum(
                1 + log_σ - keras.backend.square(μ) - keras.backend.exp(log_σ),
                axis=-1
            )
        )
        return loss + KL_divergence

    auto.compile(
        loss=KL_loss, optimizer='adam'
    )

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
    encoder, decoder, auto = autoencoder(784, [512], 2)
    auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
             validation_data=(x_test, x_test))
    encoded, mu, log_sig = encoder.predict(x_test[:10])
    print(mu)
    print(np.exp(log_sig / 2))
    reconstructed = decoder.predict(encoded).reshape((-1, 28, 28))
    x_test = x_test.reshape((-1, 28, 28))

    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        ax.axis('off')
        plt.imshow(x_test[i])
        ax = plt.subplot(2, 10, i + 11)
        ax.axis('off')
        plt.imshow(reconstructed[i])
    plt.show()

    l1 = np.linspace(-3, 3, 25)
    l2 = np.linspace(-3, 3, 25)
    L = np.stack(np.meshgrid(l1, l2, sparse=False, indexing='ij'), axis=2)
    G = decoder.predict(L.reshape((-1, 2)), batch_size=125)

    for i in range(25*25):
        ax = plt.subplot(25, 25, i + 1)
        ax.axis('off')
        plt.imshow(G[i].reshape((28, 28)))
    plt.show()
