#!/usr/bin/env python3
"""Module that contains the function crop that preforms a random crop of an
image."""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def crop_image(image, size):
    """Function that performs a random crop of an image.

    Args:
        image (tensorflow.Tensor): 3D Tensor containing an image.
        size (int, int): Tuple containing the size of the crop.

    Returns: (cropped)
        cropped (tensorflow.Tensor): 3D Tensor containing an image.
    """
    pass


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(1)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(crop_image(image, (200, 200, 3)))
        plt.show()
