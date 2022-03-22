#!/usr/bin/env python3
"""Module that contains the function sheer_image that randomly shears an
image."""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def shear_image(image, intensity):
    """Function that randomly shears an image.

    Args:
        image (tensorflow.Tensor): 3D Tensor containing an image.
        intensity (int): The intensity with which the image should be
            sheared.

    Returns: (sheared)
        sheared (tensorflow.Tensor): 3D Tensor containing an image.
    """
    array = tf.keras.preprocessing.image.img_to_array(image)
    sheared_array = tf.keras.preprocessing.image.random_shear(array, intensity)
    sheared = tf.keras.preprocessing.image.array_to_img(sheared_array)
    return sheared


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(3)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(shear_image(image, 50))
        plt.show()