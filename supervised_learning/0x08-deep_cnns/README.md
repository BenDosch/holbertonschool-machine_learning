# Deep Convolutional Architectures

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Inception Block](#0-inception-block)
	2. [Inception Network](#1-inception-network)
	3. [Identity Block](#2-identity-block)
	4. [Projection Block](#3-projection-block)
	5. [ResNet-50](#4-resnet-50)
	6. [Dense Block](#5-dense-block)
	7. [Transition Layer](#6-transition-layer)
	8. [DenseNet-121](#7-densenet-121)

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is a skip connection?
* What is a bottleneck layer?
* What is the Inception Network?
* What is ResNet? ResNeXt? DenseNet?
* How to replicate a network architecture by reading a journal article

## Refrences

* [Title](www.url.com "Title")
* [Convolutional Neural Networks (Course 4 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF "Convolutional Neural Networks (Course 4 of the Deep Learning Specialization)")

### GoogleNet

* [1x1 Convolutions](https://www.youtube.com/watch?v=SIpcirNNGAk "1x1 Convolutions")
* [GoogLeNet Tutorial](https://www.youtube.com/watch?v=_XF7N6rp9Jw "GoogLeNet Tutorial")
* [Review: GoogLeNet (Inception v1)— Winner of ILSVRC 2014 (Image Classification)](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7 "Review: GoogLeNet (Inception v1)— Winner of ILSVRC 2014 (Image Classification)")
* [A guide to Inception Model in Keras](https://maelfabien.github.io/deeplearning/inception/# "A guide to Inception Model in Keras")
* [Deep Learning in the Trenches: Understanding Inception Network from Scratch](https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/ "Deep Learning in the Trenches: Understanding Inception Network from Scratch")

### ResNet

* [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035 "An Overview of ResNet and its Variants")

### Task Refrenced Papers

* [Going Deeper with Convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf "Going Deeper with Convolutions (2014)")
* [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf "Deep Residual Learning for Image Recognition (2015)")
* [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf "Densely Connected Convolutional Networks")

## Tasks

List of tasks with brief descriptions of each task.

### [0. Inception Block](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/0-inception_block.py "0. Inception Block")

Write a function that builds an inception block as described in [Going Deeper with Convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf). All convolutions inside and outside the inception block should use a rectified linear activation (ReLU).

---

### [1. Inception Network](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/1-inception_network.py "1. Inception Network")

Write a function that builds the inception network as described in [Going Deeper with Convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf). You can assume the input data will have shape (224, 224, 3). All convolutions inside and outside the inception block should use a rectified linear activation (ReLU).

---

### [2. Identity Block](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/2-identity_block.py "2. Identity Block")

Write a function that builds an identity block as described in [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf). All convolutions inside the block should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively. All weights should use he normal initialization.

---

### [3. Projection Block](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/3-projection_block.py "3. Projection Block")

Write a function that builds a projection block as described in [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf). All convolutions inside the block should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively. All weights should use he normal initialization

---

### [4. ResNet-50](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/4-resnet50.py "4. ResNet-50")

Write a function that builds the ResNet-50 architecture as described in [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385.pdf). You can assume the input data will have shape (224, 224, 3). All convolutions inside and outside the blocks should be followed by batch normalization along the channels axis and a rectified linear activation (ReLU), respectively. All weights should use he normal initialization.

---

### [5. Dense Block](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/5-dense_block.py "5. Dense Block")

Write a function that builds a dense block as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf). You should use the bottleneck layers used for DenseNet-B. All weights should use he normal initialization. All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively.

---

### [6. Transition Layer](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/6-transition_layer.py "6. Transition Layer")

Write a function that builds a transition layer as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf). All weights should use he normal initialization. All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively.

---

### [7. DenseNet-121](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x08-deep_cnns/7-densenet121.py "7. DenseNet-121")

Write a function def densenet121(growth_rate=32, compression=1.0): that builds the DenseNet-121 architecture as described in [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf). You can assume the input data will have shape (224, 224, 3). All convolutions should be preceded by Batch Normalization and a rectified linear activation (ReLU), respectively. All weights should use he normal initialization.

---
