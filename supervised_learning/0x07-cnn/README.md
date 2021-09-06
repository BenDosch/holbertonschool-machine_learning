# Project name

(optional) overall project description.

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is a convolutional layer?
* What is a pooling layer?
* Forward propagation over convolutional and pooling layers
* Back propagation over convolutional and pooling layers
* How to build a CNN using Tensorflow and Keras

## Tasks
List of tasks with brief descriptions of each task.
### [0. Convolutional Forward Prop](https://github.com/BenDoschGit/holbertonschool-machine_learning/tree/main/0x07-cnn/0-conv_forward.py "0. Convolutional Forward Prop")

Write a function that performs forward propagation over a convolutional layer of a neural network. You may import numpy as np.

---
### [1. Pooling Forward Prop](https://github.com/BenDoschGit/holbertonschool-machine_learning/tree/main/0x07-cnn/1-pool_forward.py "1. Pooling Forward Prop")

Write a function that performs forward propagation over a pooling layer of a neural network. You may import numpy as np.

---
### [2. Convolutional Back Prop](https://github.com/BenDoschGit/holbertonschool-machine_learning/tree/main/0x07-cnn/2-conv_backward.py "2. Convolutional Back Prop")

Write a function that performs back propagation over a convolutional layer of a neural network. You may import numpy as np.

---
### [3. Pooling Back Prop](https://github.com/BenDoschGit/holbertonschool-machine_learning/tree/main/0x07-cnn/3-pool_backward.py "3. Pooling Back Prop")

Write a function that performs back propagation over a pooling layer of a neural network. You may import numpy as np.

---
### [4. LeNet-5 (Tensorflow)](https://github.com/BenDoschGit/holbertonschool-machine_learning/tree/main/0x07-cnn/4-lenet5.py "4. LeNet-5 (Tensorflow)")

Write a function that builds a modified version of the LeNet-5 architecture using tensorflow.

The model should consist of the following layers in order:
* Convolutional layer with 6 kernels of shape 5x5 with same padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Convolutional layer with 16 kernels of shape 5x5 with valid padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Fully connected layer with 120 nodes
* Fully connected layer with 84 nodes
* Fully connected softmax output layer with 10 nodes
All layers requiring initialization should initialize their kernels with the he_normal initialization method: tf.contrib.layers.variance_scaling_initializer()
All hidden layers requiring activation should use the relu activation function
You may import tensorflow as tf and may NOT use tf.keras.

---
### [5. LeNet-5 (Keras)](https://github.com/BenDoschGit/holbertonschool-machine_learning/tree/main/0x07-cnn/5-lenet5.py "5. LeNet-5 (Keras)")

Write a function that builds a modified version of the LeNet-5 architecture using keras.

The model should consist of the following layers in order:
* Convolutional layer with 6 kernels of shape 5x5 with same padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Convolutional layer with 16 kernels of shape 5x5 with valid padding
* Max pooling layer with kernels of shape 2x2 with 2x2 strides
* Fully connected layer with 120 nodes
* Fully connected layer with 84 nodes
* Fully connected softmax output layer with 10 nodes
All layers requiring initialization should initialize their kernels with the he_normal initialization method: tf.contrib.layers.variance_scaling_initializer()
All hidden layers requiring activation should use the relu activation function
You may import tensorflow.keras as K.

---
### [6. Summarize Like a Pro](www..com "6. Summarize Like a Pro")

A common practice in the machine learning industry is to read and review journal articles on a weekly basis. Read and write a summary of Krizhevsky et. al.‘s 2012 paper ImageNet Classification with Deep Convolutional Neural Networks. Your summary should include:
* Introduction: Give the necessary background to the study and state its purpose.
* Procedures: Describe the specifics of what this study involved.
* Results: In your own words, discuss the major findings and results.
* Conclusion: In your own words, summarize the researchers’ conclusions.
* Personal Notes: Give your reaction to the study.
Your posts should have examples and at least one picture, at the top.

---