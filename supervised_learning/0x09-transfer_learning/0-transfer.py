#!/usr/bin/env python3
"""Module containing """

import numpy as np
from numpy.core.fromnumeric import resize
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt


def preprocess_data(X, Y):
    """Function that preprocess data for the model.

    Args:
        X (numpy.ndarray): A N-dimensional arary with the shape of
            (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number
            of data points.
        Y (numpy.ndarray): A N-dimensional arary with the shape of (m,)
            containing the CIFAR 10 labels for X.

    Returns:
        X_p(numpy.ndarray): A N-dimensional containing the preprocessed X.
        Y_p(numpy.ndarray): A N-dimensional containing the preprocessed Y.
    """
    # Preprocess inputs to between -1 and 1
    X_p = K.applications.vgg19.preprocess_input(
        x=X, data_format="channels_last"
    )
    Y_p = K.utils.to_categorical(Y)

    return X_p, Y_p


def resize_images(X):
    """Takes a tensor of images as input and quadruples the size.

    Args:
        X (numpy.ndarray): Tensor containg image data

    Returns:
        processed (numpy.ndarray): Tensor containg resized image data.
    """
    resize_inputs = K.Input(shape=(32, 32, 3))
    resize_outputs = K.layers.Lambda(
        lambda x: K.backend.resize_images(
            x, height_factor=4, width_factor=4, data_format="channels_last"
        )  # quadruples height and widith
    )(resize_inputs)
    resize_model = K.Model(inputs=resize_inputs, outputs=resize_outputs)
    processed = resize_model.predict(X)
    return processed


def build_new():
    """[summary]

    Returns:
        [type]: [description]
    """
    init = K.initializers.he_normal(seed=None)
    new_input = K.Input(shape=(4, 4, 512))
    new_clasifier = K.layers.Flatten()(new_input)
    new_clasifier = K.layers.Dense(
            units=1000,  activation="relu", kernel_initializer=init
        )(new_clasifier)
    new_clasifier = K.layers.Dense(
            units=100, activation="relu", kernel_initializer=init
        )(new_clasifier)
    new_clasifier = K.layers.Dropout(0.3)(new_clasifier)
    new_clasifier = K.layers.Dense(
            units=10, activation="softmax", kernel_initializer=init
        )(new_clasifier)

    model = K.Model(inputs=new_input, outputs=new_clasifier)

    return model


def extract_features(X, sample_count, batch_size):
    """[summary]

    Args:
        X ([type]): [description]
        sample_count ([type]): [description]
        batch_size ([type]): [description]

    Returns:
        [type]: [description]
    """
    features = np.zeros(shape=(sample_count, 4, 4, 512))

    # Set up pre-trained model
    conv_base = K.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3)
    )

    features = conv_base.predict(X, verbose=True, batch_size=batch_size)

    return features


if __name__ == "__main__":
    # Load data and preprocess data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    """x_train = x_train[0:256, :, :, :]
    y_train = y_train[0:256, :]
    x_test = x_test[0:32, :, :, :]
    y_test = y_test[0:32, :]"""

    # Resize images
    resized = resize_images(x_train)
    resized_test = resize_images(x_test)

    # Extract features
    features = extract_features(
        X=resized, sample_count=int(resized.shape[0]), batch_size=128
    )
    features_test = extract_features(
        X=resized_test, sample_count=int(resized_test.shape[0]), batch_size=128
    )

    # Build transfer model
    new_model = build_new()

    # Complie model
    new_model.compile(
            loss='categorical_crossentropy', optimizer=K.optimizers.Adam(),
            metrics=['accuracy']
        )

    # Train model
    history = new_model.fit(
        x=features, y=y_train, validation_data=(features_test, y_test),
        batch_size=512, epochs=10, verbose=True
    )

    loss, accuracy = new_model.evaluate(features_test, y_test, verbose=0)

    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # Graph of accuracy and loss
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, 'g', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    # Finetune model

    # Save model
    # model.save('cifar10.h5')
