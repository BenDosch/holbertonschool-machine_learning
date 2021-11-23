#!/usr/bin/env python3
"""Module that contains """

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder.

    Args:
        input_dims (tuple(int)): A tuple of integers containing the dimensions
            of the model input.
        filters (list[int]): A list containing the number of filters for each
            convolutional layer in the encoder, respectively. The filters
            should be reversed for the decoder.
        latent_dims (int): A  tuple of integers containing the dimensions of
            the latent space representation.

    Returns: encoder, decoder, auto
        encoder(): The encoder model.
        decoder(): The decoder model.
        auto(): The full autoencoder model.
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
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    print(x_train.shape)
    print(x_test.shape)
    np.random.seed(0)
    tf.set_random_seed(0)
    encoder, decoder, auto = autoencoder((28, 28, 1), [16, 8, 8], (4, 4, 8))
    auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
                    validation_data=(x_test, x_test))
    encoded = encoder.predict(x_test[:10])
    print(np.mean(encoded))
    reconstructed = decoder.predict(encoded)[:,:,:,0]

    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        ax.axis('off')
        plt.imshow(x_test[i,:,:,0])
        ax = plt.subplot(2, 10, i + 11)
        ax.axis('off')
        plt.imshow(reconstructed[i])
    plt.show()