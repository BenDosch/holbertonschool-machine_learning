#!/usr/bin/env python3
"""Module that contains the function flip_image, whcihc flips an image
horizontally."""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def flip_image(image):
    """Function that flips an image horizontally.

    Args:
        image (tensorflow.Tensor): 3D Tensor containing an image.

    Returns: (fliped)
        fliped (tensorflow.Tensor): 3D Tensor containing an image.
    """
    pass


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(0)

    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(flip_image(image))
        plt.show()
