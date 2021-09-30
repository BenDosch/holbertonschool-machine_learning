#!/usr/bin/env python3
"""Moduel containing the class Yolo used for Object Detection."""

import tensorflow as tf
import tensorflow.keras as K


class Yolo():
    """Class that uses the Yolo v3 algorithm to perform object detection.

    Public Instance Attributes:
        model (tensorflow.keras.Model): The Darknet Keras model.
        class_names (list): A list of the class names for the model.
        class_t (float): The box score threshold for the initial filtering
            step.
        nms_t (float): The IOU threshold for non-max suppression.
        anchors (numpy.ndarray): The anchor boxes.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initilization method for the class YOLO.

        Args:
            model_path (str): The path where the Darknet
                Keras model is stored.
            class_path (str): The path to where the list of class names used
                for the Darknet model, listed in order of index, can be found.
            class_t (float): A float representing the box score threshold for
                the initial filtering step.
            nms_t (float): A float representing the IOU threshold for non-max
                suppression.
            anchors (numpy.ndarray): Tensor of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes. Where outputs is the number
                of outputs (predictions) made by the Darknet model,
                anchor_boxes is the number of anchor boxes used for each
                prediction, and 2 is [anchor_box_width, anchor_box_height].
        """
        model = K.models.load_model(model_path)
        class_names = []
        with open(classes_path, 'r') as file:
            for line in file:
                class_names.append(line.strip())
        class_names = class_names
        self.model = model
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
