from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def run_mnist_softmax(sess, mnist):
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        W = tf.Variable(tf.zeros([784,10]))
        b = tf.Variable(tf.zeros([10]))
        sess.run(tf.global_variables_initializer())
        y = tf.matmul(x,W) + b
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        for _ in range(50):
            batch = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


def run_mnist_conv(sess, mnist):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    return sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


class TestMnistConv(unittest.TestCase):

    def test_cpu(self):
        tf.reset_default_graph()
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        tf.set_random_seed(233)
        np.random.seed(233)

        with tf.device('/device:CPU:0'):
            with tf.Session() as sess:
                expected = run_mnist_softmax(sess, mnist)


    def test_softmax(self):
        tf.reset_default_graph()
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        tf.set_random_seed(233)
        np.random.seed(233)
        with tf.device('/device:RPC:0'):
            with tf.Session() as sess:
                actual = run_mnist_softmax(sess, mnist)

        tf.reset_default_graph()
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        tf.set_random_seed(233)
        np.random.seed(233)

        with tf.device('/device:CPU:0'):
            with tf.Session() as sess:
                expected = run_mnist_softmax(sess, mnist)

        self.assertEquals(actual, expected)

    def test_conv(self):
        tf.reset_default_graph()
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        tf.set_random_seed(233)
        np.random.seed(233)
        with tf.device('/device:RPC:0'):
            with tf.Session() as sess:
                actual = run_mnist_conv(sess, mnist)

        tf.reset_default_graph()
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        tf.set_random_seed(233)
        np.random.seed(233)

        with tf.device('/device:CPU:0'):
            with tf.Session() as sess:
                expected = run_mnist_conv(sess, mnist)

        self.assertEquals(actual, expected)


if __name__ == '__main__':
    unittest.main()
