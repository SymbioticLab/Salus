from __future__ import print_function

import unittest

import numpy as np
from datetime import datetime
from timeit import default_timer

from parameterized import parameterized

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from . import run_on_rpc_and_cpu, run_on_devices, run_on_sessions, assertAllClose
from .lib import tfhelper


def run_mnist_softmax(sess, mnist, batch_size=50):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.global_variables_initializer())
    y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    for _ in range(20):
        batch = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


def run_mnist_conv(sess, mnist, batch_size=50):
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
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fc2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_fc2, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_num = 20
    speeds = []
    with tfhelper.initialized_scope(sess) as coord:
        for i in range(batch_num):
            if coord.should_stop():
                break
            batch = mnist.train.next_batch(batch_size)
            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            duration = default_timer() - start_time
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            speeds.append(sec_per_batch)
            loss_value = sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch'
            print(fmt_str.format(datetime.now(), i, loss_value, examples_per_sec, sec_per_batch))
        print('Average %.3f sec/batch' % np.average(speeds))

        return sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


def run_mnist_large(sess, mnist, batch_size=50):
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
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_conv4 = weight_variable([5, 5, 64, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    h_pool_flat = tf.reshape(h_pool4, [-1, 512])

    W_fc1 = weight_variable([512, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc11 = weight_variable([1024, 1024])
    b_fc11 = bias_variable([1024])
    h_fc11 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc11) + b_fc11)
    h_fc11_drop = tf.nn.dropout(h_fc11, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_fc2 = tf.matmul(h_fc11_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_fc2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_fc2, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_num = 20
    speeds = []
    inbetween = []
    last_end_time = 0
    with tfhelper.initialized_scope(sess) as coord:
        JCT = default_timer()
        for i in range(batch_num):
            if coord.should_stop():
                break

            batch = mnist.train.next_batch(batch_size)
            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            _, loss_value = sess.run([train_step, cross_entropy],
                                    feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            end_time = default_timer()

            if last_end_time > 0:
                inbetween.append(start_time - last_end_time)
            last_end_time = end_time

            duration = end_time - start_time
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            speeds.append(sec_per_batch)

            fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch'
            print(fmt_str.format(datetime.now(), i, loss_value, examples_per_sec, sec_per_batch))
        print('Average %.3f sec/batch' % np.average(speeds))
        print('Average %.6f sec spent between batches' % np.average(inbetween))
        JCT = default_timer() - JCT
        print('Training time is %.3f sec' % JCT)

        print('Start final eva')
        start_time = default_timer()
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        duration = default_timer() - start_time
        print('Final eval takes %.3f sec' % duration)
        return acc


class MnistConvBase(unittest.TestCase):
    def _runner(self):
        return None

    def _config(self, **kwargs):
        return tf.ConfigProto()

    @parameterized.expand([(25,), (50,), (100,)])
    def test_gpu(self, batch_size):
        def func():
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            sess = tf.get_default_session()
            return self._runner()(sess, mnist, batch_size=batch_size)

        run_on_devices(func, '/device:GPU:0', config=self._config(batch_size=batch_size))

    def test_cpu(self):
        def func():
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            sess = tf.get_default_session()
            return self._runner()(sess, mnist)

        run_on_devices(func, '/device:CPU:0', config=self._config())

    @parameterized.expand([(25,), (50,), (100,)])
    def test_rpc(self, batch_size):
        def func():
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            sess = tf.get_default_session()
            return self._runner()(sess, mnist, batch_size=batch_size)
        run_on_sessions(func, 'zrpc://tcp://127.0.0.1:5501', config=self._config(batch_size=batch_size))

    def test_correctness(self):
        def func():
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            sess = tf.get_default_session()
            return self._runner()(sess, mnist)

        actual, expected = run_on_rpc_and_cpu(func, config=self._config())
        assertAllClose(actual, expected, rtol=1e-3)


class TestMnistSoftmax(MnistConvBase):
    def _runner(self):
        return run_mnist_softmax


class TestMnistConv(MnistConvBase):
    def _runner(self):
        return run_mnist_conv


class TestMnistLarge(MnistConvBase):
    def _runner(self):
        return run_mnist_large

    def _config(self, **kwargs):
        memusages = {
            25: (5715962884 - 23415216, 23415216),
            50: (5715962884 - 23494616, 23494616),
            100: (5715962884 - 23653416, 23653416),
        }
        batch_size = kwargs.get('batch_size', 50)

        config = tf.ConfigProto()
        config.zmq_options.resource_map.temporary['MEMORY:GPU'] = memusages[batch_size][0]
        config.zmq_options.resource_map.persistant['MEMORY:GPU'] = memusages[batch_size][1]
        return config


del MnistConvBase


if __name__ == '__main__':
    unittest.main()
