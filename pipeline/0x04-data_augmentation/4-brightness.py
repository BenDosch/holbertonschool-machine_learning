#!/usr/bin/env python3
"""Module that contains the function change_brightness that randomly changes
the brightness of an image."""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def change_brightness(image, max_delta):
    """Function that randomly changes the brightness of an image.

    Args:
        image (tensorflow.Tensor): 3D Tensor containing an image.
        max_delta (float): The maximum amount the image should be brightened
            (or darkened).
    
    Returns: (altered)
        altered (tensorflow.Tensor): 3D Tensor containing an image.
    """
    altered = tf.image.random_brightness(image, max_delta)
    return altered


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(4)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(change_brightness(image, 0.3))
        plt.show()
