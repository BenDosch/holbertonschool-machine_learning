#!/usr/bin/env python3
"""Module that contains the function rotate_image that rotates an image by 90
degrees counter-clockwise."""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def rotate_image(image):
    """Function that rotates an image by 90 degrees counter-clockwise.

    Args:
        image (tensorflow.Tensor): 3D Tensor containing an image.

    Returns: (rotated)
        rotated (tensorflow.Tensor): 3D Tensor containing an image.
    """
    pass


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(2)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(rotate_image(image))
        plt.show()
