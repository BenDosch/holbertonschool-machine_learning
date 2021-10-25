#!/usr/bin/env python3
"""Module that contains the function model.
"""

import tensorflow as tf
create_placeholders = __import__('../0x02-tensorflow/0-create_placeholders').create_placeholders
forward_prop = __import__('../0x02-tensorflow/2-forward_prop').forward_prop
calculate_accuracy = __import__('../0x02-tensorflow/3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('../0x02-tensorflow/4-calculate_loss').calculate_loss
shuffle_data = __import__('2-shuffle_data').shuffle_data
create_Adam_op = __import__('10-Adam').create_Adam_op


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """Function that builds, trains, and saves a neural network model in
    tensorflow using Adam optimization, mini-batch gradient descent, learning
    rate decay, and batch normalization.

    Args:
        Data_train (tuple): Tuple containing the training inputs and training
            lables respectively.
        Data_valid (tuple): Tuple containing the training inputs and validation
            lables respectively.
        layers (list[(int)]): List containing the number of nodes in each layer.
        activations (list([type])): List containing the activation functions
            used for each layer of the network.
        alpha (float, optional): The learning rate. Defaults to 0.001.
        beta1 (float, optional): The weight for the first moment of Adam
            Optimization. Defaults to 0.9.
        beta2 (float, optional): The weight for the second moment of Adam
            Optimization. Defaults to 0.999.
        epsilon ([type], optional): A small number to avoid division by zero.
            Defaults to 1e-8.
        decay_rate (int, optional): The decay rate for invers time decay of the
            learning rate. Defaults to 1.
        batch_size (int, optional): The number of data points that should be in
            a mini-batch. Defaults to 32.
        epochs (int, optional): The number of times the training should pass
            through the whole dataset. Defaults to 5.
        save_path (str, optional): The path where the model should be saved to.
            Defaults to '/tmp/model.ckpt'.

    Returns:
        The path where the model was saved.
    """
    X_train, Y_train = Data_train[0], Data_train[1]
    X_valid, Y_valid = Data_valid[0], Data_valid[1]
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    train_op = create_Adam_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    store = tf.train.Saver()

    with tf.Session as sess:

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