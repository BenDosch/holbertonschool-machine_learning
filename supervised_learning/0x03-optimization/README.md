# Optimization

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
    1. [Normalization Constants](#0-normalization-constants)
    2. [Normalize](#1-normalize)
    3. [Shuffle Data](#2-shuffle-data)
    4. [Mini-Batch](#3-mini-batch)
    5. [Moving Average](#4-moving-average)
    6. [Momentum](#5-momentum)
    7. [Momentum Upgraded](#6-momentum-upgraded)
    8. [RMSProp](#7-rmsprop)
    9. [RMSProp Upgraded](#8-rmsprop-upgraded)
    10. [Adam](#9-adam)
    11. [Adam Upgraded](#10-adam-upgraded)
    12. [Learning Rate Decay](#11-learning-rate-decay)
    13. [Learning Rate Decay Upgraded](#12-learning-rate-decay-upgraded)
    14. [Batch Normalization](#13-batch-normalization)
    15. [Batch Normalization Upgraded](#14-batch-normalization-upgraded)
    16. [Put it all together and what do you get?](#15-put-it-all-together-and-what-do-you-get?)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is a hyperparameter?
* How and why do you normalize your input data?
* What is a saddle point?
* What is stochastic gradient descent?
* What is mini-batch gradient descent?
* What is a moving average? How do you implement it?
* What is gradient descent with momentum? How do you implement it?
* What is RMSProp? How do you implement it?
* What is Adam optimization? How do you implement it?
* What is learning rate decay? How do you implement it?
* What is batch normalization? How do you implement it?

## References

* [Tensorflow](https://github.com/tensorflow/docs/ "Tensorflow")
* [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/ "A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size")
* [Why, How and When to Scale your Features](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e "Why, How and When to Scale your Features")
* [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/ "An overview of gradient descent optimization algorithms")
* [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Course 2 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Course 2 of the Deep Learning Specialization)")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Normalization Constants](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/0-norm_constants.py "0. Normalization Constants")

Write the function that calculates the normalization (standardization) constants of a matrix.

---
### [1. Normalize](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/1-normalize.py "1. Normalize")

Write a function that normalizes (standardizes) a matrix.

---
### [2. Shuffle Data](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/2-shuffle_data.py "2. Shuffle Data")

Write a function that shuffles the data points in two matrices the same way.

---
### [3. Mini-Batch](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/3-mini_batch.py "3. Mini-Batch")

Write a function that trains a loaded neural network model using mini-batch gradient descent.

---
### [4. Moving Average](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/4-moving_average.py "4. Moving Average")

Write a function that calculates the weighted moving average of a data set.

---
### [5. Momentum](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/5-momentum.py "5. Momentum")

Write a function that updates a variable using the gradient descent with momentum optimization algorithm.

---
### [6. Momentum Upgraded](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/6-momentum.py "6. Momentum Upgraded")

Write a function that creates the training operation for a neural network in tensorflow using the gradient descent with momentum optimization algorithm.

---
### [7. RMSProp](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/7-RMSProp.py "7. RMSProp")

Write a function that updates a variable using the RMSProp optimization algorithm.

---
### [8. RMSProp Upgraded](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/8-RMSProp.py "8. RMSProp Upgraded")

Write a function that creates the training operation for a neural network in tensorflow using the RMSProp optimization algorithm.

---
### [9. Adam](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/9-Adam.py "9. Adam")

Write a function that updates a variable in place using the Adam optimization algorithm.

---
### [10. Adam Upgraded](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/10-Adam.py "10. Adam Upgraded")

Write a function that creates the training operation for a neural network in tensorflow using the Adam optimization algorithm.

---
### [11. Learning Rate Decay](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/11-learning_rate_decay.py "11. Learning Rate Decay")

Write a function def learning_rate_decay(alpha, decay_rate, global_step, decay_step): that updates the learning rate using inverse time decay in numpy. The learning rate decay should occur in a stepwise fashion.

---
### [12. Learning Rate Decay Upgraded](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/12-learning_rate_decay.py "12. Learning Rate Decay Upgraded")

Write a function that creates a learning rate decay operation in tensorflow using inverse time decay. The learning rate decay should occur in a stepwise fashion.

---
### [13. Batch Normalization](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/13-batch_norm.py "13. Batch Normalization")

Write a function that normalizes an unactivated output of a neural network using batch normalization.

---
### [14. Batch Normalization Upgraded](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/14-batch_norm.py "14. Batch Normalization Upgraded")

Write a function that creates a batch normalization layer for a neural network in tensorflow.

---
### [15. Put it all together and what do you get?](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x03-optimization/15-model.py "15. Put it all together and what do you get?")

Write a function that builds, trains, and saves a neural network model in tensorflow using Adam optimization, mini-batch gradient descent, learning rate decay, and batch normalization.
    
Note: the input data does not need to be normalized as it has already been scaled to a range of [0, 1]

---
