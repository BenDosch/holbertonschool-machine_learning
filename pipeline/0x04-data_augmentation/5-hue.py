#!/usr/bin/env python3
"""Module that contains the function change_hue that that changes the hue of
an image."""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def change_hue(image, delta):
    """Function that changes the hue of an image.

    Args:
        image (tensorflow.Tensor): 3D Tensor containing an image.
        delta (float): The amount the hue should change.
    
    Returns: (altered)
        altered (tensorflow.Tensor): 3D Tensor containing an image.
    """
    altered = tf.image.adjust_hue(image, delta)
    return altered


if __name__ == "__main__":
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(5)
    
    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(change_hue(image, -0.5))
        plt.show()
