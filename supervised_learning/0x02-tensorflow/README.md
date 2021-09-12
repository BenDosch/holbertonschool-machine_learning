# Tensorflow

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
    1. [Placeholders](#0-placeholders)
    2. [Layers](#1-layers)
    3. [Forward Propagation](#2-forward-propagation)
    4. [Accuracy](#3-accuracy)
    5. [Loss](#4-loss)
    6. [train_op](#5-train_op)
    7. [Train](#6-train)
    8. [Evaluate](#7-Evaluate)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is tensorflow?
* What is a session? graph?
* What are tensors?
* What are variables? constants? placeholders? How do you use them?
* What are operations? How do you use them?
* What are namespaces? How do you use them?
* How to train a neural network in tensorflow
* What is a checkpoint?
* How to save/load a model with tensorflow
* What is the graph collection?
* How to add and get variables from the collection

## References
* [Tensorflow](https://github.com/tensorflow/docs/ "Tensorflow")
* [W3clubDocs / Tensorflow Python](https://docs.w3cub.com/tensorflow~python/ "W3clubDocs / Tensorflow Python")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Placeholders](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/0-create_placeholders.py "0. Placeholders")

Write a function that performs forward propagation over a convolutional layer of a neural network. You may import numpy as np.

---
### [1. Layers](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/1-create_layer.py "1. Layers")

Write a function that applies an activation function to the output of a tensorflow layer.

---
### [2. Forward Propagation](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/2-forward_prop.py "2. Forward Propagation")

Write a function  that creates the forward propagation graph for the neural network.

---
### [3. Accuracy](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/3-calculate_accuracy.py "3. Accuracy")

Write a function that calculates the accuracy of a prediction.

---
### [4. Loss](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/4-calculate_loss.py "4. Loss")

Write a function that calculates the softmax cross-entropy loss of a prediction.

---
### [5. Train_Op](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/5-create_train_op.py "5. Train_Op")

Write a function that creates the training operation for the network.

---
### [6. Train](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/6-train.py "6. Train")

Write a function that builds, trains, and saves a neural network classifier.

---
### [7. Evaluate](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x02-tensorflow/7-evaluate.py "7. Evaluate")

Write a function that evaluates the output of a neural network. You are not allowed to use tf.saved_model

---