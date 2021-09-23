#!/usr/bin/env python3
"""Module containing several functions related to setting up a transfer
learning model. The main() function trains and saves a model to classify the
CIFAR10 dataset utilizing transfer learning"""

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
    X_p = K.applications.xception.preprocess_input(
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


def build_new_classifyer(input_shape):
    """Builds a new model to replace the old clasificatin steps of a transfered
    model.

    Args:
        input_shape (tuple): shape of input for new model.

    Returns:
        Model: Model to preform classification
    """
    init = K.initializers.he_normal(seed=None)
    new_input = K.Input(shape=input_shape)
    new_clasifier = K.layers.Dense(
            units=1000,  activation="relu", kernel_initializer=init
        )(new_input)
    new_clasifier = K.layers.Dropout(0.5)(new_clasifier)
    new_clasifier = K.layers.Dense(
            units=100, activation="relu", kernel_initializer=init
        )(new_clasifier)
    new_clasifier = K.layers.Dropout(0.5)(new_clasifier)
    new_clasifier = K.layers.Dense(
            units=10, activation="softmax", kernel_initializer=init
        )(new_clasifier)

    model = K.Model(inputs=new_input, outputs=new_clasifier)

    return model


def extract_features(X, batch_size):
    """Gets the output of a base model and returns the output of the final
    layers as a numpy.ndarray.

    Args:
        X (numpy.ndarray): Input tensor with shape (m, 128, 128, 3) where
            m is the number of samples.
        batch_size (int): The size of each batch.

    Returns:
        numpy.ndarray: The output of the final layer of the model.
    """
    conv_base = K.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3),
        pooling="avg"
    )
    # conv_base.summary()

    return  conv_base.predict(X, verbose=True, batch_size=batch_size)



def graph_loss_accuray(history):
    """Takes a History object and graphs the loss and accuray of the training
    and validation across epochs.

    Args:
        history (History): A record of training loss values and metrics values
            at successive epochs, as well as validation loss values and
            validation metrics values.
    """
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
    x_train = resize_images(x_train)
    x_test = resize_images(x_test)

    # Extract features
    x_train_extract = extract_features(
        X=x_train[0:32, :, :, :], batch_size=32
    )
    """x_test_extract = extract_features(
        X=x_test, batch_size=32
    )"""

    # Build transfer model
    new_model = build_new_classifyer(input_shape=x_train_extract.shape[1:])

    # Complie model
    new_model.compile(
            loss='categorical_crossentropy', optimizer=K.optimizers.Adam(),
            metrics=['accuracy']
        )

    # Train model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    """history = new_model.fit(
        x=x_train_extract, y=y_train, validation_data=(x_test_extract, y_test),
        batch_size=512, epochs=15, callbacks=[callback], verbose=True
    )"""

    # Graph of accuracy and loss
    # graph_loss_accuray(history=history)

    # Finetune model
    base_model = K.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3),
        pooling="avg"
    )
    X = new_model(base_model.output)
    model = K.Model(inputs=base_model.inputs, outputs=X)
    model.compile(
        loss='categorical_crossentropy', optimizer=K.optimizers.Adam(),
        metrics=['accuracy']
    )
    history = model.fit(
        x=x_train, y=y_train, validation_data=(x_test, y_test),
        batch_size=128, epochs=2, callbacks=[callback], verbose=True
    )

    # graph_loss_accuray(history=history)

    # Save model
    model.save('cifar10.h5')
