# Keras

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
    1. [Sequential](#0-sequential)
    2. [Input](#1-input)
    3. [Optimize](#2-optimize)
    4. [One Hot](#3-one_hot)
    5. [Train](#4-train)
    6. [Validate](#5-validate)
    7. [Early Stopping](#6-early-Stopping)
    8. [Learning Rate Decay](#7-learning-rate-decay)
    9. [Save Only the Best](#8-save-only-the-best)
    10. [Save and Load Model](#9-save-and-load-model)
    11. [Save and Load Weights](#10-save-and-load-weights)
    12. [Save and Load Configuration](#11-save-and-load-configuration)
    13. [Test](#12-test)
    14. [Predict](#13-predict)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is Keras?
* What is a model?
* How to instantiate a model (2 ways)
* How to build a layer
* How to add regularization to a layer
* How to add dropout to a layer
* How to add batch normalization
* How to compile a model
* How to optimize a model
* How to fit a model
* How to use validation data
* How to perform early stopping
* How to measure accuracy
* How to evaluate a model
* How to make a prediction with a model
* How to access the weights/outputs of a model
* What is HDF5?
* How to save and load a model’s weights, a model’s configuration, and the entire model

## References
* [Keras Explained](https://www.youtube.com/watch?v=j_pJmXJwMLA&t=228s "Keras Explained")
* [Keras](https://www.tensorflow.org/guide/keras "Keras")

## Tasks
### [0. Sequential](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/0-sequential.py "0. Sequential")

Write a function that builds a neural network with the Keras library. You are not allowed to use the Input class.

---
### [1. Input](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/1-input.py "1. Input")

Write a function that builds a neural network with the Keras library. You are not allowed to use the Sequential class.

---
### [2. Optimize](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/2-optimize.py "2. Optimize")

Write a function that sets up Adam optimization for a keras model with categorical crossentropy loss and accuracy metrics.

---
### [3. One Hot](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/3-one_hot.py "3. One Hot")

Write a function that converts a label vector into a one-hot matrix.

---
### [4. Train](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/4-train.py "4. Train")

Write a function that trains a model using mini-batch gradient descent.

---
### [5. Validate](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/5-train.py "5. Validate")

Based on 4-train.py, update the function to also analyze validaiton data.

---
### [6. Early Stopping](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/6-train.py "6. Early Stopping")

Based on 5-train.py, update the function to also train the model using early stopping.

---
### [7. Learning Rate Decay](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/7-train.py "7. Learning Rate Decay")

Based on 6-train.py, update the function to also train the model with learning rate decay.

---
### [8. Save Only the Best](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/8-train.py "8. Save Only the Best")

Based on 7-train.py, update the function to also save the best iteration of the model.

---
### [9. Save and Load Model](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/9-model.py "9. Save and Load Model")

Write the following functions:

save_model: saves an entire model.
load_model: loads an entire model.

---
### [10. Save and Load Weights](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/10-weights.py "10. Save and Load Weights")

Write the following functions:

save_weights: saves a model’s weights.
load_weights: loads a model’s weights.

---
### [11. Save and Load Configuration](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/11-config.py "11. Save and Load Configuration")

Write the following functions:

save_config: saves a model’s configuration in JSON format.
load_config: loads a model with a specific configuration.

---
### [12. Test](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/12-test.py "12. Test")

Write a function that tests a neural network.

---
### [13. Predict](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x06-keras/13-predict.py "13. Predict")

Write a function that makes a prediction using a neural network.

---
