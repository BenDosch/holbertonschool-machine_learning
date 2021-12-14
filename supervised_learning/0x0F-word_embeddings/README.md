# Word_Embeddings

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Bag Of Words](#0-bag-of-words)
	2. [TF-IDF](#1-tf-idf)
	3. [Train Word2Vec](#2-train-word2vec)
	4. [Extract Word2Vec](#3-extract-word2vec)
	5. [FastText](#4-fasttext)
	6. [ELMo](#5-elmo)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google.

* What is natural language processing?
* What is a word embedding?
* What is bag of words?
* What is TF-IDF?
* What is CBOW?
* What is a skip-gram?
* What is an n-gram?
* What is negative sampling?
* What is word2vec, GloVe, fastText, ELMo?

## Refrences

* [An Introduction to Word Embeddings](https://www.springboard.com/blog/data-science/introduction-word-embeddings/ "An Introduction to Word Embeddings")
*[Introduction to Word Embeddings](http://hunterheidenreich.com/blog/intro-to-word-embeddings/ "Introduction to Word Embeddings")
* [Natural language Processing](https://www.youtube.com/playlist?list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm "Natural language Processing")
* [Word Embedding - Natural Language Processing| Deep Learning](https://www.youtube.com/watch?v=pO_6Jk0QtKw "Word Embedding - Natural Language Processing| Deep Learning")
* [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/ "Word2Vec Tutorial - The Skip-Gram Model")
* [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/ Word2Vec Tutorial Part 2 - Negative Sampling)
* [Word Vectors in Natural Language Processing: Global Vectors (GloVe)](https://medium.com/sciforce/word-vectors-in-natural-language-processing-global-vectors-glove-51339db89639 "Word Vectors in Natural Language Processing: Global Vectors (GloVe)")
* [FastText: Under the Hood](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3 "FastText: Under the Hood")
* [Deep Contextualized Word Representations with ELMo](https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/ "Deep Contextualized Word Representations with ELMo")
* [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer "sklearn.feature_extraction.text.CountVectorizer")
* [Gensim API Reference](https://radimrehurek.com/gensim/apiref.htmlhtml#pretrained-models "Gensim API Reference")
* [Using pretrained gensim Word2vec embedding in keras](https://stackoverflow.com/questions/52126539/using-pretrained-gensim-word2vec-embedding-in-keras "Using pretrained gensim Word2vec embedding in keras")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Bag Of Words](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x0F-word_embeddings/0-bag_of_words.py "0. Bag Of Words")

Write a function def bag_of_words(sentences, vocab=None): that creates a bag of words embedding matrix.

* sentences is a list of sentences to analyze
* vocab is a list of the vocabulary words to use for the analysis
	* If None, all words within sentences should be used
* Returns: embeddings, features
	* embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
		* s is the number of sentences in sentences
		* f is the number of features analyzed
	* features is a list of the features used for embeddings

---

### [1. TF-IDF](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x0F-word_embeddings/1-tf_idf.py "1. TF-IDF")

Write a function def tf_idf(sentences, vocab=None): that creates a TF-IDF embedding:

* sentences is a list of sentences to analyze
* vocab is a list of the vocabulary words to use for the analysis
	* If None, all words within sentences should be used
* Returns: embeddings, features
	* embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
		* s is the number of sentences in sentences
		* f is the number of features analyzed
	* features is a list of the features used for embeddings

---

### [2. Train Word2Vec](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x0F-word_embeddings/2-word2vec.py "2. Train Word2Vec")

Write a function def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a gensim word2vec model.

* sentences is a list of sentences to be trained on
* size is the dimensionality of the embedding layer
* min_count is the minimum number of occurrences of a word for use in training
* window is the maximum distance between the current and predicted word within a sentence
* negative is the size of negative sampling
* cbow is a boolean to determine the training type; True is for CBOW; False is for Skip-gram
* iterations is the number of iterations to train over
* seed is the seed for the random number generator
* workers is the number of worker threads to train the model
* Returns: the trained model.

---

### [3. Extract Word2Vec](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x0F-word_embeddings/3-gensim_to_keras.py "3. Extract Word2Vec")

Write a function def gensim_to_keras(model): that converts a gensim word2vec model to a keras Embedding layer.

* model is a trained gensim word2vec models
* Returns: the trainable keras Embedding

---

### [4. FastText](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x0F-word_embeddings/4-fasttext.py "4. FastText")

Write a function def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a genism fastText model.

* sentences is a list of sentences to be trained on
* size is the dimensionality of the embedding layer
* min_count is the minimum number of occurrences of a word for use in training
* window is the maximum distance between the current and predicted word within a sentence
* negative is the size of negative sampling
* cbow is a boolean to determine the training type; True is for CBOW; False is for Skip-gram
* iterations is the number of iterations to train over
* seed is the seed for the random number generator
* workers is the number of worker threads to train the model
* Returns: the trained model

---

### [5. ELMo](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/0x0F-word_embeddings/5-elmo "5. ELMo")

When training an ELMo embedding model, you are training.

1. The internal weights of the BiLSTM
2. The character embedding layer
3. The weights applied to the hidden states

In the text file 5-elmo, write the letter answer, followed by a newline, that lists the correct statements:

* A. 1, 2, 3
* B. 1, 2
* C. 2, 3
* D. 1, 3
* E. 1
* F. 2
* G. 3
* H. None of the above

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
