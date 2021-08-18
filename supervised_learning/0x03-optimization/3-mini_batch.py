#!/usr/bin/env python3
"""Module that contains the function train_mini_batch.
"""

import tensorflow as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Function that trains a loaded neural network model using mini-batch
        gradient descent.

    Args:
        X_train (numpy.ndarray): N-dimensional array with the shape of (m, 784)
            containing the training data.
        Y_train (numpy.ndarray): N-dimensional array with the shape of (m, 10)
            containing the training data.
        X_valid (numpy.ndarray): N-dimensional array with the shape of (m, 784)
            containing the validation lables.
        Y_valid (numpy.ndarray): N-dimensional array with the shape of (m, 10)
            containing the validation lables.
        batch_size (int, optional): The number of data points in a batch.
            Defaults to 32.
        epochs (int, optional): The number of times the training should pass
            through the whole dataset. Defaults to 5.
        load_path (str, optional): The path from which to load the model.
            Defaults to "/tmp/model.ckpt".
        save_path (str, optional): The path to where the model should be saved
            after training. Defaults to "/tmp/model.ckpt".
        
    Returns: the path where the model was saved
    """
    shuffle_data = __import__('2-shuffle_data').shuffle_data
    sess = tf.Session()
    saver = tf.train.import_meta_graph(load_path + '.meta')
    store = tf.train.Saver()
    saver.restore(sess, load_path)
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    accuracy = tf.get_collection('accuracy')[0]
    loss = tf.get_collection('loss')[0]
    train_op = tf.get_collection('train_op')[0]

    for epoch in range(epochs + 1):
        train = sess.run([loss, accuracy], feed_dict={x: X_train, y: Y_train})
        valid = sess.run([loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
        
        print("After {} epochs:".format(epoch))
        print("\tTraining Cost: {}".format(train[0]))
        print("\tTraining Accuracy: {}".format(train[1]))
        print("\tValidation Cost: {}".format(valid[0]))
        print("\tValidation Accuracy: {}".format(valid[1]))

        if epoch < epochs:
            X_train_s, Y_train_s = shuffle_data(X_train, Y_train)
            m = X_train.shape[0]

            if m % batch_size == 0:
                total_batches = m // batch_size
            else:
                total_batches = (m // batch_size) + 1

            for batch in range(total_batches):
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                step = batch + 1

                if batch_end > m:
                    X_t = X_train[batch_start:, :]
                    Y_t = Y_train_s[batch_start:, :]
                else:
                    X_t = X_train_s[batch_start:batch_end, :]
                    Y_t = Y_train_s[batch_start:batch_end, :]
                results = sess.run([accuracy, loss, train_op],
                                    feed_dict={x: X_t, y: Y_t})
                
                if step % 100 == 0:
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(results[1]))
                    print("\t\tAccuracy: {}".format(results[0]))               

    return store.save(sess, save_path)
