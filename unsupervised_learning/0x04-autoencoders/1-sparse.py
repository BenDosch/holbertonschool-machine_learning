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
        encoder(): The encoder model.
        decoder(): The decoder model.
        auto(): The sparse autoencoder model.
    """
    pass


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
