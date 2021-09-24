# Transfer Learning

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Transfer Knowledge](#0-transfer-knowledge)
	2. ["Research is what I'm doing when I don't know what I'm doing." - Wernher von Braun](#1-research)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is a transfer learning?
* What is fine-tuning?
* What is a frozen layer? How and why do you freeze a layer?
* How to use transfer learning with Keras applications

## Refrences

* [Keras Applications](https://keras.io/api/applications/ "Keras Applications")
* [Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/ "Transfer learning & fine-tuning")
* [A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a "A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning")
* [Transfer learning / fine-tuning](https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb#scrollTo=zMxC6Pd1YobN "Transfer learning / fine-tuning")
*[How to Extract Features from Images?](https://www.youtube.com/watch?v=PaSEVY9d4RI "How to Extract Features from Images?")
* [Convolutional Neural Networks (Course 4 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF "Convolutional Neural Networks (Course 4 of the Deep Learning Specialization)")
* [Change input shape dimensions for fine-tuning with Keras](https://www.pyimagesearch.com/2019/06/24/change-input-shape-dimensions-for-fine-tuning-with-keras/ "Change input shape dimensions for fine-tuning with Keras")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Transfer Knowledge](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/g/0-transfer.py "0. Transfer Knowledge")

Write a python script that trains a convolutional neural network to classify the [CIFAR 10 dataset](https://keras.io/api/datasets/cifar10/ "CIFAR 10 dataset").

* You must use one of the applications listed in Keras Applications
* Your script must save your trained model in the current working directory as cifar10.h5
* Your saved model should be compiled
* Your saved model should have a validation accuracy of 87% or higher
* Your script should not run when the file is imported
* Hint1: The training and tweaking of hyperparameters may take a while so start early!
* Hint2: The CIFAR 10 dataset contains 32x32 pixel images, however most of the Keras applications are trained on much larger images. Your first layer should be a lambda layer that scales up the data to the correct size
* Hint3: You will want to freeze most of the application layers. Since these layers will always produce the same output, you should compute the output of the frozen layers ONCE and use those values as input to train the remaining trainable layers. This will save you A LOT of time.

In the same file, write a function that pre-processes the data for your model.

---

### [1. "Research is what I'm doing when I don't know what I'm doing." - Wernher von Braun](https://medium.com/@BenDosch/first-experience-with-transfer-learning-7f54e1ec786d "First experience with transfer learning.")

Write a blog post explaining your experimental process in completing the task above written as a journal-style scientific paper. Your posts should have examples and at least one picture, at the top.

---