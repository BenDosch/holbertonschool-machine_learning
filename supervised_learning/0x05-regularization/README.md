# Directory

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
    1. [L2 Regularization Cost](#0-l2-regularization-cost)
    2. [Gradient Descent with L2 Regularization](#1-gradient-descent-with-l2-regularization)
    3. [L2 Regularization Cost](#2-l2-regularization-cost)
    4. [Create a Layer with L2 Regularization](#3-create-a-layer-with-l2-regularization)
    5. [Forward Propagation with Dropout](#4-forward-propagation-with-dropout)
    6. [Gradient Descent with Dropout](#5-gradient-descent-with-dropout)
    7. [Create a Layer with Dropout](#6-create-a-layer-with-dropout)
    8. [Early Stopping](#7-early-stopping)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is regularization? What is its purpose?
* What is are L1 and L2 regularization? What is the difference between the two methods?
* What is dropout?
* What is early stopping?
* What is data augmentation?
* How do you implement the above regularization methods in Numpy? Tensorflow?
* What are the pros and cons of the above regularization methods?

## References
* [https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/](https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/ "An Overview of Regularization Techniques in Deep Learning (with Python code)")
* [L2 Regularization and Back-Propagation](https://jamesmccaffrey.wordpress.com/2017/02/19/l2-regularization-and-back-propagation/ "L2 Regularization and Back-Propagation")
* [Early stopping](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf "Early stopping")
* [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Course 2 of the Deep Learning Specialization)](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc "Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization (Course 2 of the Deep Learning Specialization)")

## Tasks

### [0. L2 Regularization Cost](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/0-l2_reg_cost.py "0. L2 Regularization Cost")

Write a function that calculates the cost of a neural network with L2 regularization.

---
### [1. Gradient Descent with L2 Regularization](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/1-l2_reg_gradient_descent.py "1. Gradient Descent with L2 Regularization")

Write a function that updates the weights and biases of a neural network using gradient descent with L2 regularization.

---
### [2. L2 Regularization Cost](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/2-l2_reg_cost.py "2. L2 Regularization Cost")

Write the function that calculates the cost of a neural network with L2 regularization.

---
### [3. Create a Layer with L2 Regularization](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/3-l2_reg_create_layer.py "3. Create a Layer with L2 Regularization")

Write a function that creates a tensorflow layer that includes L2 regularization.

---
### [4. Forward Propagation with Dropout](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/4-dropout_forward_prop.py "4. Forward Propagation with Dropout")

Write a function that conducts forward propagation using Dropout.

---
### [5. Gradient Descent with Dropout](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/5-dropout_gradient_descent.py "5. Gradient Descent with Dropout")

Write a function that updates the weights of a neural network with Dropout regularization using gradient descent. The weights of the network should be updated in place.

---
### [6. Create a Layer with Dropout](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/6-dropout_create_layer.py "6. Create a Layer with Dropout")

Write a function that creates a layer of a neural network using dropout.

---
### [7. Early Stopping](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x05-regularization/7-early_stopping.py "7. Early Stopping")

Write a function that determines if you should stop gradient descent early.

---
