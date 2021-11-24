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
        latent_dims (tuple(int)): A tuple of integers containing the dimensions
            of the latent space representation.

    Returns: encoder, decoder, auto
        encoder(keras.models.Model): The encoder model.
        decoder(keras.models.Model): The decoder model.
        auto(keras.models.Model): The full autoencoder model.
    """
    kernel_size = (3, 3)
    filter_len = len(filters)

    # Encoder
    enco_in = keras.Input(shape=input_dims)

    enco_hid = keras.layers.Conv2D(
        filters=filters[0], kernel_size=kernel_size, activation='relu',
        padding='same'
    )(enco_in)
    enco_hid = keras.layers.MaxPool2D(
                pool_size=(2, 2), padding="same"
            )(enco_hid)
    if filter_len > 1:
        for i in range(1, filter_len):
            enco_hid = keras.layers.Conv2D(
                filters=filters[i], kernel_size=kernel_size, activation='relu',
                padding='same'
            )(enco_hid)
            enco_hid = keras.layers.MaxPool2D(
                pool_size=(2, 2), padding="same"
            )(enco_hid)

    encoder = keras.models.Model(inputs=enco_in, outputs=enco_hid)

    # Decoder
    deco_in = keras.Input(shape=latent_dims)

    for j in range(filter_len - 1, -1, -1):
        if j == 0:
            padding = 'valid'
        else:
            padding = 'same'

        if j == filter_len - 1:
            deco_hid = keras.layers.Conv2D(
                filters=filters[j], kernel_size=kernel_size, activation='relu',
                padding=padding
            )(deco_in)
        else:
            deco_hid = keras.layers.Conv2D(
                filters=filters[j], kernel_size=kernel_size, activation='relu',
                padding=padding
            )(deco_hid)

        deco_hid = keras.layers.UpSampling2D(size=(2, 2))(deco_hid)

    deco_out = keras.layers.Conv2D(
        filters=input_dims[-1], kernel_size=kernel_size, activation='sigmoid',
        padding='same'
    )(deco_hid)

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
    reconstructed = decoder.predict(encoded)[:, :, :, 0]

    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        ax.axis('off')
        plt.imshow(x_test[i, :, :, 0])
        ax = plt.subplot(2, 10, i + 11)
        ax.axis('off')
        plt.imshow(reconstructed[i])
    plt.show()
