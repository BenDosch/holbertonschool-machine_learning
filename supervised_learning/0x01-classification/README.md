# Classification

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
    1. [Neuron](#0-neuron)
    2. [Privatize Neuron](#1-privatize-neuron)
    3. [Neuron Forward Propagation](#2-neuron-forward-propagation)
    4. [Neuron Cost](#3-neuron-cost)
    5. [Evaluate Neuron](#4-evaluate-neuron)
    6. [Neuron Gradient Descent](#5-neuron-gradient-descent)
    7. [Train Neuron](#6-train-neuron)
    8. [Upgrade Train Neuron](#7-upgrade-train-neuron)
    9. [NeuralNetwork](#8-neuralnetwork)
    10. [Privatize NeuralNetwork](#9-privatize-neuralnetwork)
    11. [NeuralNetwork Forward Propagation](#10-neuralnetwork-forward-propagation)
    12. [NeuralNetwork Cost](#11-neuralnetwork-cost)
    13. [Evaluate NeuralNetwork](#12-evaluate-neuralnetwork)
    14. [NeuralNetwork Gradient Descent](#13-neuralnetwork-gradient-descent)
    15. [Train NeuralNetwork](#14-train-neuralnetwork)
    16. [Upgrade Train NeuralNetwork](#15-upgrade-train-neuralnetwork)
    17. [DeepNeuralNetwork](#16-deepneuralnetwork)
    18. [Privatize DeepNeuralNetwork](#17-privatize-deepneuralnetwork)
    19. [DeepNeuralNetwork Forward Propagation](#18-deepneuralnetwork-forward-propagation)
    20. [DeepNeuralNetwork Cost](#19-deepneuralnetwork-cost)
    21. [Evaluate DeepNeuralNetwork](#20-evaluate-deepneuralnetwork)
    22. [DeepNeuralNetwork Gradient Descent](#21-deepneuralnetwork-gradient-descent)
    23. [Train DeepNeuralNetwork](#22-train-deepneuralnetwork)
    24. [Upgrade Train DeepNeuralNetwork](#23-upgrade-train-deepneuralnetwork)
    25. [One-Hot Encode](#24-one-hot-encode)
    26. [One-Hot Decode](#25-one-hot-decode)
    27. [Persistence is Key](#26-persistence-is-key)
    28. [Update DeepNeuralNetwork](#27-update-deepneuralnetwork)
    29. [All the Activations](#28-all-the-activations)
    20. [Blogpost](#29-blogpost)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is a model?
* What is supervised learning?
* What is a prediction?
* What is a node?
* What is a weight?
* What is a bias?
* What are activation functions?
    * Sigmoid?
    * Tanh?
    * Relu?
    * Softmax?
* What is a layer?
* What is a hidden layer?
* What is Logistic Regression?
* What is a loss function?
* What is a cost function?
* What is forward propagation?
* What is Gradient Descent?
* What is back propagation?
* What is a Computation Graph?
* How to initialize weights/biases
* The importance of vectorization
* How to split up your data
* What is multiclass classification?
* What is a one-hot vector?
* How to encode/decode one-hot vectors
* What is the softmax function and when do you use it?
* What is cross-entropy loss?
* What is pickling in Python?

## References

* [DeepAI series: Intro to neural networks](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0 "DeepAI series: Intro to neural networks")
* [DeepAI series: Optimizing deep nerual networks](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc "DeepAI series: Optimizing deep nerual networks")
* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6 "Activation Functions in Neural Networks")
* [Softmax as Activation Function](https://www.python-course.eu/softmax.php "Softmax as Activation Function")
* [Neural network initialization](https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e "Neural network initialization")
* [Machine Learning Glossary: Neural Networks](https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html "Machine Learning Glossary: Neural Networks")
[Supervised vs. Unsupervised Machine Learning](https://blogs.oracle.com/ai-and-datascience/post/supervised-vs-unsupervised-machine-learning "Supervised vs. Unsupervised Machine Learning")
* [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html "Using neural nets to recognize handwritten digits")
* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/ "Calculus on Computational Graphs: Backpropagation")
* [Derivation: Derivatives for Common Neural Network Activation Functions](https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions "Derivation: Derivatives for Common Neural Network Activation Functions")
* [What is One Hot Encoding? Why and When Do You Have to Use it?](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f?gi=a4f47cf027f7 "What is One Hot Encoding? Why and When Do You Have to Use it?")
* [NumPy v1.18 Manual](https://numpy.org/doc/1.18/index.html "NumPy v1.18 Manual")
* [Pickle](https://docs.python.org/3/library/pickle.html "Pickle")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Neuron](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/0-neuron.py "0. Neuron")

Write a class Neuron that defines a single neuron performing binary classification.

* class constructor: def __init__(self, nx):
    * nx is the number of input features to the neuron
        * If nx is not an integer, raise a TypeError with the exception: nx must be an integer
        * If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
    * All exceptions should be raised in the order listed above
    * Public instance attributes:
    * W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
    * b: The bias for the neuron. Upon instantiation, it should be initialized to 0.
    * A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.

---
### [1. Privatize Neuron](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/1-neuron.py "1. Privatize Neuron")

Write a class Neuron that defines a single neuron performing binary classification (Based on 0-neuron.py).

* Private instance attributes:
    * __W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
    * __b: The bias for the neuron. Upon instantiation, it should be initialized to 0.
    * __A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.
    * Each private attribute should have a corresponding getter function (no setter function).

---
### [2. Neuron Forward Propagation](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/2-neuron.py "2. Neuron Forward Propagation")

Write a class Neuron that defines a single neuron performing binary classification (Based on 1-neuron.py).

* Add the public method def forward_prop(self, X):
    * Calculates the forward propagation of the neuron
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Updates the private attribute __A
    * The neuron should use a sigmoid activation function
    * Returns the private attribute __A

---
### [3. Neuron Cost](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/3-neuron.py "3. Neuron Cost")

Write a class Neuron that defines a single neuron performing binary classification (Based on 2-neuron.py).

* Add the public method def cost(self, Y, A):
    * Calculates the cost of the model using logistic regression
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
    * To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
    * Returns the cost

---
### [4. Evaluate Neuron](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/4-neuron.py "4. Evaluate Neuron")

Write a class Neuron that defines a single neuron performing binary classification (Based on 3-neuron.py).

* Add the public method def evaluate(self, X, Y):
    * Evaluates the neuron’s predictions
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * Returns the neuron’s prediction and the cost of the network, respectively
        * The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
        * The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

---
### [5. Neuron Gradient Descent](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/5-neuron.py "5. Neuron Gradient Descent")

Write a class Neuron that defines a single neuron performing binary classification (Based on 4-neuron.py).

* Add the public method def gradient_descent(self, X, Y, A, alpha=0.05):
    * Calculates one pass of gradient descent on the neuron
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
    * alpha is the learning rate
    * Updates the private attributes __W and __b

---
### [6. Train Neuron](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/6-neuron.py "6. Train Neuron")

Write a class Neuron that defines a single neuron performing binary classification (Based on 5-neuron.py).

* Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
    * Trains the neuron
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * iterations is the number of iterations to train over
        * if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
        * if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
        * alpha is the learning rate
            * if alpha is not a float, raise a TypeError with the exception alpha must be a float
            * if alpha is not positive, raise a ValueError with the exception alpha must be positive
    * All exceptions should be raised in the order listed above
    * Updates the private attributes __W, __b, and __A
    * You are allowed to use one loop
    * Returns the evaluation of the training data after iterations of training have occurred

---
### [7. Upgrade Train Neuron](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/7-neuron.py "7. Upgrade Train Neuron")

Write a class Neuron that defines a single neuron performing binary classification (Based on 6-neuron.py)

* Update the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
    *  Trains the neuron by updating the private attributes __W, __b, and __A
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * iterations is the number of iterations to train over
        * if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
        * if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
    * alpha is the learning rate
        * if alpha is not a float, raise a TypeError with the exception alpha must be a float
        * if alpha is not positive, raise a ValueError with the exception alpha must be positive
    * verbose is a boolean that defines whether or not to print information about the training. If True, print Cost after {iteration} iterations: {cost} every step iterations:
        * Include data from the 0th and last iteration
    * graph is a boolean that defines whether or not to graph information about the training once the training has completed. If True:
    * Plot the training data every step iterations as a blue line
    * Label the x-axis as iteration
    * Label the y-axis as cost
    * Title the plot Training Cost
    * Include data from the 0th and last iteration
    * Only if either verbose or graph are True:
        * if step is not an integer, raise a TypeError with the exception step must be an integer
        * if step is not positive or is greater than iterations, raise a ValueError with the exception step must be positive and <= iterations
    * All exceptions should be raised in the order listed above
    * The 0th iteration should represent the state of the neuron before any training has occurred
    * You are allowed to use one loop
    * You can use import matplotlib.pyplot as plt
    * Returns: the evaluation of the training data after iterations of training have occurred

---
### [8. NeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/8-neural_network.py "8. NeuralNetwork")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification.

* class constructor: def __init__(self, nx, nodes):
    * nx is the number of input features
        * If nx is not an integer, raise a TypeError with the exception: nx must be an integer
        * If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
    * nodes is the number of nodes found in the hidden layer
        * If nodes is not an integer, raise a TypeError with the exception: nodes must be an integer
        * If nodes is less than 1, raise a ValueError with the exception: nodes must be a positive integer
    * All exceptions should be raised in the order listed above
* Public instance attributes:
    * W1: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
    *  b1: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
    * A1: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
    * W2: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
    * b2: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
    * A2: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.

---
### [9. Privatize NeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/9-neural_network.py "9. Privatize NeuralNetwork")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 8-neural_network.py).

*  Private instance attributes:
    * W1: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
    * b1: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
    * A1: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
    * W2: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
    *  b2: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
    * A2: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.
    * Each private attribute should have a corresponding getter function (no setter function).

---
### [10. NeuralNetwork Forward Propagation](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/10-neural_network.py "10. NeuralNetwork Forward Propagation")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 9-neural_network.py).

* Add the public method def forward_prop(self, X):
    * Calculates the forward propagation of the neural network
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Updates the private attributes __A1 and __A2
    * The neurons should use a sigmoid activation function
    * Returns the private attributes __A1 and __A2, respectively

---
### [11. NeuralNetwork Cost](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/11-neural_network.py "11. NeuralNetwork Cost")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 10-neural_network.py).

* Add the public method def cost(self, Y, A):
    * Calculates the cost of the model using logistic regression
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
    * To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
    * Returns the cost

---
### [12. Evaluate NeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/12-neural_network.py "12. Evaluate NeuralNetwork")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 11-neural_network.py).

* Add the public method def evaluate(self, X, Y):
    * Evaluates the neural network’s predictions
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * Returns the neuron’s prediction and the cost of the network, respectively
    * The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
    * The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise.

---
### [13. NeuralNetwork Gradient Descent](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/13-neural_network.py "13. NeuralNetwork Gradient Descent")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 12-neural_network.py).

* Add the public method def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
    * Calculates one pass of gradient descent on the neural network
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * A1 is the output of the hidden layer
    * A2 is the predicted output
    * alpha is the learning rate
    * Updates the private attributes __W1, __b1, __W2, and __b2

---
### [14. Train NeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/14-neural_network.py "14. Train NeuralNetwork")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 13-neural_network.py).

* Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
    * Trains the neural network
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * iterations is the number of iterations to train over
        * if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
        * if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
    * alpha is the learning rate
        * if alpha is not a float, raise a TypeError with the exception alpha must be a float
        * if alpha is not positive, raise a ValueError with the exception alpha must be positive
    * All exceptions should be raised in the order listed above
    * Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2
    * You are allowed to use one loop
    * Returns the evaluation of the training data after iterations of training have occurred

---
### [15. Upgrade Train NeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/15-neural_network.py "15. Upgrade Train NeuralNetwork")

Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 14-neural_network.py).

* Update the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
    * Trains the neural network
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * iterations is the number of iterations to train over
        * if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
        * if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
    * alpha is the learning rate
        * if alpha is not a float, raise a TypeError with the exception alpha must be a float
        * if alpha is not positive, raise a ValueError with the exception alpha must be positive
    * Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2
    * verbose is a boolean that defines whether or not to print information about the training. If True, print Cost after {iteration} iterations: {cost} every step iterations:
        * Include data from the 0th and last iteration
    * graph is a boolean that defines whether or not to graph information about the training once the training has completed. If True:
    * Plot the training data every step iterations as a blue line
    * Label the x-axis as iteration
    * Label the y-axis as cost
    * Title the plot Training Cost
    * Include data from the 0th and last iteration
    * Only if either verbose or graph are True:
        * if step is not an integer, raise a TypeError with the exception step must be an integer
        * if step is not positive and less than or equal to iterations, raise a ValueError with the exception step must be positive and <= iterations
    * All exceptions should be raised in the order listed above
    * The 0th iteration should represent the state of the neuron before any training has occurred
    * You are allowed to use one loop
    * Returns the evaluation of the training data after iterations of training have occurred

---
### [16. DeepNeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/16-deep_neural_network.py "16. DeepNeuralNetwork")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification.

* class constructor: def __init__(self, nx, layers):
    * nx is the number of input features
        * If nx is not an integer, raise a TypeError with the exception: nx must be an integer
        * If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
    * layers is a list representing the number of nodes in each layer of the network
        * If layers is not a list or an empty list, raise a TypeError with the exception: layers must be a list of positive integers
        * The first value in layers represents the number of nodes in the first layer, …
        * If the elements in layers are not all positive integers, raise a TypeError with the exception layers must be a list of positive integers
    * All exceptions should be raised in the order listed above
    * Sets the public instance attributes:
        * L: The number of layers in the neural network.
        * cache: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.
        * weights: A dictionary to hold all weights and biased of the network. Upon instantiation:
            * The weights of the network should be initialized using the He et al. method and saved in the weights dictionary using the key W{l} where {l} is the hidden layer the weight belongs to
            * The biases of the network should be initialized to 0’s and saved in the weights dictionary using the key b{l} where {l} is the hidden layer the bias belongs to
    * You are allowed to use one loop

---
### [17. Privatize DeepNeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/17-deep_neural_network.py "17. Privatize DeepNeuralNetwork")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 22-deep_neural_network.py).

* Sets the private instance attributes:
    * __L: The number of layers in the neural network.
    * __cache: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.
    * __weights: A dictionary to hold all weights and biased of the network. Upon instantiation:
        * The weights of the network should be initialized using the He et al. method and saved in the __weights dictionary using the key W{l} where {l} is the hidden layer the weight belongs to
        * The biases of the network should be initialized to 0‘s and saved in the __weights dictionary using the key b{l} where {l} is the hidden layer the bias belongs to
    * Each private attribute should have a corresponding getter function (no setter function).

---
### [18. DeepNeuralNetwork Forward Propagation](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/18-deep_neural_network.py "18. DeepNeuralNetwork Forward Propagation")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 17-deep_neural_network.py).

* Add the public method def forward_prop(self, X):
    * Calculates the forward propagation of the neural network
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Updates the private attribute __cache:
        * The activated outputs of each layer should be saved in the __cache dictionary using the key A{l} where {l} is the hidden layer the activated output belongs to
        * X should be saved to the cache dictionary using the key A0
    * All neurons should use a sigmoid activation function
    * Returns the output of the neural network and the cache, respectively

---
### [19. DeepNeuralNetwork Cost](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/19-deep_neural_network.py "19. DeepNeuralNetwork Cost")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 18-deep_neural_network.py).

* Add the public method def cost(self, Y, A):
    * Calculates the cost of the model using logistic regression
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
    * To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
    * Returns the cost

---
### [20. Evaluate DeepNeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/20-deep_neural_network.py "20. Evaluate DeepNeuralNetwork")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 19-deep_neural_network.py).

* Add the public method def evaluate(self, X, Y):
    * Evaluates the neural network’s predictions
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * Returns the neuron’s prediction and the cost of the network, respectively
        * The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
        * The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

---
### [21. DeepNeuralNetwork Gradient Descent](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/21-deep_neural_network.py "21. DeepNeuralNetwork Gradient Descent")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 20-deep_neural_network.py).

* Add the public method def gradient_descent(self, Y, cache, alpha=0.05):
    * Calculates one pass of gradient descent on the neural network
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * cache is a dictionary containing all the intermediary values of the network
    * alpha is the learning rate
    * Updates the private attribute __weights
    * You are allowed to use one loop

---
### [22. Train DeepNeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/22-deep_neural_network.py "22. Train DeepNeuralNetwork")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 21-deep_neural_network.py).

* Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
    * Trains the deep neural network
    *  X is a  numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * iterations is the number of iterations to train over
        * if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
        * if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
    * alpha is the learning rate
        * if alpha is not a float, raise a TypeError with the exception alpha must be a float
        * if alpha is not positive, raise a ValueError with the exception alpha must be positive
    * All exceptions should be raised in the order listed above
    * Updates the private attributes __weights and __cache
    * You are allowed to use one loop
    * Returns the evaluation of the training data after iterations of training have occurred

---
### [23. Upgrade Train DeepNeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/23-deep_neural_network.py "23. Upgrade Train DeepNeuralNetwork")

Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 22-deep_neural_network.py).

* Update the public method train to def train(self, X, Y, iterations=5000,  alpha=0.05, verbose=True, graph=True, step=100):
    * Trains the deep neural network by updating the private attributes __weights and __cache
    * X is a numpy.ndarray with shape (nx, m) that contains the input data
        * nx is the number of input features to the neuron
        * m is the number of examples
    * Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    * iterations is the number of iterations to train over
        * if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
        * if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
    * alpha is the learning rate
        * if alpha is not a float, raise a TypeError with the exception alpha must be a float
        * if alpha is not positive, raise a ValueError with the exception alpha must be positive
    * verbose is a boolean that defines whether or not to print information about the training. If True, print Cost after {iteration} iterations: {cost} every step iterations:
        * Include data from the 0th and last iteration
    * graph is a boolean that defines whether or not to graph information about the training once the training has completed. If True:
        * Plot the training data every step iterations as a blue line
        * Label the x-axis as iteration
        * Label the y-axis as cost
        * Title the plot Training Cost
        * Include data from the 0th and last iteration
    * Only if either verbose or graph are True:
        * if step is not an integer, raise a TypeError with the exception step must be an integer
    * All exceptions should be raised in the order listed above
    * The 0th iteration should represent the state of the neuron before any training has occurred
    * You are allowed to use one loop
    * Returns the evaluation of the training data after iterations of training have occurred

---
### [24. One-Hot Encode](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/24-one_hot_encode.py "24. One-Hot Encode")

Write a function that converts a numeric label vector into a one-hot matrix.

---
### [25. One-Hot Decode](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/25-one_hot_decode.py "25. One-Hot Decode")

Write a function that converts a one-hot matrix into a vector of labels.

---
### [26. Persistence is Key](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/26-deep_neural_network.py "26. Persistence is Key")

Update the class DeepNeuralNetwork (based on 23-deep_neural_network.py) with a instance method to save the instance object to a file in pickle format, and the static method to load a pickled DeepNeuralNetwork object.

---
### [27. Update DeepNeuralNetwork](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/27-deep_neural_network.py "27. Update DeepNeuralNetwork")

Update the class DeepNeuralNetwork to perform multiclass classification (based on 26-deep_neural_network.py)

* You will need to update the instance methods forward_prop, cost, and evaluate
* Y is now a one-hot numpy.ndarray of shape (classes, m)

---
### [28. All the Activations](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x01-classification/28-deep_neural_network.py "28. All the Activations")

Update the class DeepNeuralNetwork to allow different activation functions (based on 27-deep_neural_network.py).

* Update the __init__ method to def __init__(self, nx, layers, activation='sig'):
    * activation represents the type of activation function used in the hidden layers
        * sig represents a sigmoid activation
        * tanh represents a tanh activation
        * if activation is not sig or tanh, raise a ValueError with the exception: activation must be 'sig' or 'tanh'
    * Create the private attribute __activation and set it to the value of activation
    * Create a getter for the private attribute __activation
    * Update the forward_prop and gradient_descent instance methods to use the __activation function in the hidden layers

---
### [29. Blogpost](https://medium.com/@BenDosch/ml-activation-functions-f851fd6334d2 "29. Blogpost")

Write a blog post that explains the purpose of activation functions and compares and contrasts (at the minimum) the following functions:

* Binary
* Linear
* Sigmoid
* Tanh
* ReLU
* Softmax

---
