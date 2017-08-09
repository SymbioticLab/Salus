'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from parameterized import parameterized, param

from . import run_on_rpc_and_cpu, assertAllClose


def run_linear_reg(sess, training_epochs=100, learning_rate=0.01):
    # Training Data
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])

    n_samples = train_X.shape[0]

    # tf Graph Input
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # Set model weights
    W = tf.Variable(np.random.randn(), name="weight")
    b = tf.Variable(np.random.randn(), name="bias")

    # Construct a linear model
    pred = tf.add(tf.multiply(X, W), b)

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = opt.compute_gradients(cost)
    train_op = opt.apply_gradients(grads_and_vars)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    sess.run(init)

    # collect
    preds = []
    preds.append(sess.run([pred, W, b], feed_dict={X: 5.6}))

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            gv = sess.run(grads_and_vars, feed_dict={X: x, Y: y})
            sess.run(train_op, feed_dict={X: x, Y: y})
            p = sess.run(pred, feed_dict={X: x, Y: y})
            preds.append([p] + gv)

    preds.append(sess.run([pred, W, b], feed_dict={X: 5.6}))
    return preds


class TestLinreg(unittest.TestCase):
    @parameterized.expand((2**i,) for i in range(4,5))
    def test_linear_regression(self, epochs):
        def func():
            sess = tf.get_default_session()
            return run_linear_reg(sess, training_epochs=epochs)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)
