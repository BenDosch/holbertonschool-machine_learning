# Autoencoders

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. ["Vanilla" Autoencoder](#0-"vanilla"-autoencoder)
	2. [Sparse Autoencoder](#1-sparse-autoencoder)
	3. [Convolutional Autoencoder](#2-convolutional-autoencoder)
	4. [Variational Autoencoder](#3-variational-autoencoder)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is an autoencoder?
* What is latent space?
* What is a bottleneck?
* What is a sparse autoencoder?
* What is a convolutional autoencoder?
* What is a generative model?
* What is a variational autoencoder?
* What is the Kullback-Leibler divergence?

## Refrences

* [Autoencoder - definition](https://www.youtube.com/watch?v=FzS3tMl4Nsc&t=73s "Autoencoder - definition")
* [Autoencoder - loss function](https://www.youtube.com/watch?v=xTU79Zs4XKY "Autoencoder - loss function")
* [Deep learning - deep autoencoder](https://www.youtube.com/watch?v=z5ZYm_wJ37c "Deep learning - deep autoencoder")
* [Introduction to autoencoders](https://www.jeremyjordan.me/autoencoders/ "Introduction to autoencoders")
* [Variational Autoencoders - EXPLAINED! up to 12:55](https://www.youtube.com/watch?v=fcvYpzHmhvA "Variational Autoencoders - EXPLAINED! up to 12:55")
* [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8 "Variational Autoencoders")
* [Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf?gi=35aa97896522 "Intuitively Understanding Variational Autoencoders")
* [Deep Generative Models up to Generative Adversarial Networks](https://towardsdatascience.com/deep-generative-models-25ab2821afd3 "Deep Generative Models up to Generative Adversarial Networks")

## Tasks
List of tasks with brief descriptions of each task.

### [0. "Vanilla" Autoencoder](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x04-autoencoders/0-vanilla.py "0. "Vanilla" Autoencoder")

Function def autoencoder(input_dims, hidden_layers, latent_dims): that creates an autoencoder.

* input_dims is an integer containing the dimensions of the model input
* hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
* the hidden layers should be reversed for the decoder
* latent_dims is an integer containing the dimensions of the latent space representation
* Returns: encoder, decoder, auto
	* encoder is the encoder model
	* decoder is the decoder model
	* auto is the full autoencoder model
* The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
* All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid

---

### [1. Sparse Autoencoder](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x04-autoencoders/1-sparse.py "1. Sparse Autoencoder")

Write a function def autoencoder(input_dims, hidden_layers, latent_dims, lambtha): that creates a sparse autoencoder.

* input_dims is an integer containing the dimensions of the model input
* hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
	* the hidden layers should be reversed for the decoder
* latent_dims is an integer containing the dimensions of the latent space representation
* lambtha is the regularization parameter used for L1 regularization on the encoded output
* Returns: encoder, decoder, auto
	* encoder is the encoder model
	* decoder is the decoder model
	* auto is the sparse autoencoder model
* The sparse autoencoder model should be compiled using adam optimization and binary cross-entropy loss
* All layers should use a relu activation except for the last layer in the decoder, which should use sigmoid

---

### [2. Convolutional Autoencoder](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x04-autoencoders/2-convolutional.py "2. Convolutional Autoencoder")

Write a function def autoencoder(input_dims, filters, latent_dims): that creates a convolutional autoencoder.

* input_dims is a tuple of integers containing the dimensions of the model input
* filters is a list containing the number of filters for each convolutional layer in the encoder, respectively
	* the filters should be reversed for the decoder
* latent_dims is a tuple of integers containing the dimensions of the latent space representation
* Each convolution in the encoder should use a kernel size of (3, 3) with same padding and relu activation, followed by max pooling of size (2, 2)
* Each convolution in the decoder, except for the last two, should use a filter size of (3, 3) with same padding and relu activation, followed by upsampling of size (2, 2)
	* The second to last convolution should instead use valid padding
	* The last convolution should have the same number of filters as the number of channels in input_dims with sigmoid activation and no upsampling
* Returns: encoder, decoder, auto
	* encoder is the encoder model
	* decoder is the decoder model
	* auto is the full autoencoder model
* The autoencoder model should be compiled using adam optimization and binary cross-entropy loss

---

### [3. Variational Autoencoder](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x04-autoencoders/3-variational.py "3. Variational Autoencoder")

Write a function def autoencoder(input_dims, hidden_layers, latent_dims): that creates a variational autoencoder.

* input_dims is an integer containing the dimensions of the model input
* hidden_layers is a list containing the number of nodes for each hidden layer in the encoder, respectively
	* the hidden layers should be reversed for the decoder
* latent_dims is an integer containing the dimensions of the latent space representation
* Returns: encoder, decoder, auto
	* encoder is the encoder model, which should output the latent representation, the mean, and the log variance, respectively
	* decoder is the decoder model
	* auto is the full autoencoder model
* The autoencoder model should be compiled using adam optimization and binary cross-entropy loss
* All layers should use a relu activation except for the mean and log variance layers in the encoder, which should use None, and the last layer in the decoder, which should use sigmoid

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
