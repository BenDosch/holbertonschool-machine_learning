#!/usr/bin/env python3
"""Moduel containing the class Yolo used for Object Detection."""

import tensorflow as tf
import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """Processses the output of the Darkent Model.

        Args:
            outputs (list[numpy.ndarray]): A list containing the predictions
                from the Darknet model for a single image.
            image_size (numpy.ndarray): A tensor containing the image’s
                original size [image_height, image_width].

        Returns:
            A tuple of (boxes, box_confidences, box_class_probs).
            boxes (list): A List of tensors of shape (grid_height, grid_width,
                anchor_boxes, 4) containing the processed boundary boxes for
                each output, respectively. Where 4 is (x1, y1, x2, y2), and
                (x1, y1, x2, y2) should represent the boundary box relative to
                original image.
            box_confidences (list): A list of tensors of shape (grid_height,
                grid_width, anchor_boxes, 1) containing the box confidences for
                each output, respectively.
            box_class_probs (list): A list of tensors of shape (grid_height,
                grid_width, anchor_boxes, classes) containing the box’s class
                probabilities for each output, respectively.
        """
        def sigmoid(array):
            """Sigmoid activation function"""
            return 1 / (1 + np.exp(-1 * array))

        boxes, box_confidences, box_class_probs = [], [], []
        image_width = self.model.input.shape[1].value
        image_height = self.model.input.shape[2].value

        for i, output in enumerate(outputs):
            output_boxes = output[..., :4]
            grid_height, grid_width, anchors = output.shape[:3]

            cx = np.arange(grid_width).reshape(1, grid_width)
            cx = np.repeat(cx, grid_height, axis=0)
            cx = np.repeat(cx[..., np.newaxis], anchors, axis=2)
            cy = np.arange(grid_width).reshape(1, grid_width)
            cy = np.repeat(cy, grid_height, axis=0).T
            cy = np.repeat(cy[..., np.newaxis], anchors, axis=2)

            tx = output_boxes[..., 0]
            ty = output_boxes[..., 1]
            tw = output_boxes[..., 2]
            th = output_boxes[..., 3]

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bx = (sigmoid(tx) + cx) / grid_width
            by = (sigmoid(ty) + cy) / grid_height
            bw = (pw * np.exp(tw)) / image_width
            bh = (ph * np.exp(th)) / image_height

            x1 = (bx - (bw / 2)) * image_size[1]
            y1 = (by - (bh / 2)) * image_size[0]
            x2 = (bx + (bw / 2)) * image_size[1]
            y2 = (by + (bh / 2)) * image_size[0]

            output_boxes[..., 0] = x1
            output_boxes[..., 1] = y1
            output_boxes[..., 2] = x2
            output_boxes[..., 3] = y2

            boxes.append(output_boxes)
            box_confidences.append(sigmoid(output[..., 4]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return (boxes, box_confidences, box_class_probs)
