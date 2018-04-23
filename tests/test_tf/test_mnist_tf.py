from __future__ import print_function

import unittest

import numpy as np
from datetime import datetime
from timeit import default_timer

from parameterized import parameterized

import tensorflow as tf

from . import run_on_rpc_and_gpu, run_on_devices, run_on_sessions, assertAllClose
from .lib import tfhelper
from .lib.datasets import fake_data


def run_mnist_softmax(sess, batch_size=50):
    x_image, y_, num_classes = fake_data(batch_size, None, height=28, width=28, depth=1, num_classes=10)
    y_ = tf.one_hot(y_, num_classes)
    x = tf.reshape(x_image, [-1, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_), logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    with tfhelper.initialized_scope(sess) as coord:
        jct = default_timer()
        speeds = []
        losses = []
        for i in range(tfhelper.iteration_num_from_env()):
            if coord.should_stop():
                break

            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            _, loss_value, _ = sess.run([train_step, cross_entropy, tf.random_normal([1], name="salus_main_iter")])
            duration = default_timer() - start_time
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            speeds.append(sec_per_batch)

            losses.append(loss_value)
            fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
            print(fmt_str.format(datetime.now(), i, loss_value, examples_per_sec, sec_per_batch))
        jct = default_timer() - jct
        print('Training time is %.3f sec' % jct)
        print('Average: %.3f sec/batch' % np.average(speeds))
        if len(speeds) > 1:
            print('First iteration: %.3f sec/batch' % speeds[0])
            print('Average excluding first iteration: %.3f sec/batch' %
                  np.average(speeds[1:]))

        return losses


def run_mnist_conv(sess, batch_size=50):
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

    x_image, y_, num_classes = fake_data(batch_size, None, height=28, width=28, depth=1, num_classes=10)
    y_ = tf.one_hot(y_, num_classes)
    keep_prob = tf.placeholder(tf.float32)

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
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_), logits=y_fc2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    speeds = []
    losses = []
    with tfhelper.initialized_scope(sess) as coord:
        jct = default_timer()
        for i in range(tfhelper.iteration_num_from_env()):
            if coord.should_stop():
                break
            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            sess.run([train_step, tf.random_normal([1], name="salus_main_iter")], feed_dict={keep_prob: 0.5})
            duration = default_timer() - start_time
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            speeds.append(sec_per_batch)
            loss_value = sess.run(cross_entropy, feed_dict={keep_prob: 0.5})

            losses.append(loss_value)
            fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
            print(fmt_str.format(datetime.now(), i, loss_value, examples_per_sec, sec_per_batch))
        jct = default_timer() - jct
        print('Training time is %.3f sec' % jct)
        print('Average: %.3f sec/batch' % np.average(speeds))
        if len(speeds) > 1:
            print('First iteration: %.3f sec/batch' % speeds[0])
            print('Average excluding first iteration: %.3f sec/batch' %
                  np.average(speeds[1:]))

        return losses


def run_mnist_large(sess, batch_size=50):
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

    x_image, y_, num_classes = fake_data(batch_size, None, height=28, width=28, depth=1, num_classes=10)
    y_ = tf.one_hot(y_, num_classes)
    keep_prob = tf.placeholder(tf.float32)

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
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(y_), logits=y_fc2))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    speeds = []
    inbetween = []
    last_end_time = 0
    losses = []
    with tfhelper.initialized_scope(sess) as coord:
        jct = default_timer()
        for i in range(tfhelper.iteration_num_from_env()):
            if coord.should_stop():
                break

            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            _, loss_value, _ = sess.run([train_step, cross_entropy, tf.random_normal([1], name="salus_main_iter")],
                                        feed_dict={keep_prob: 0.5})
            end_time = default_timer()

            if last_end_time > 0:
                inbetween.append(start_time - last_end_time)
            last_end_time = end_time

            duration = end_time - start_time
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            speeds.append(sec_per_batch)

            losses.append(loss_value)
            fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
            print(fmt_str.format(datetime.now(), i,
                                 loss_value, examples_per_sec, sec_per_batch))
        jct = default_timer() - jct
        print('Training time is %.3f sec' % jct)
        print('Average: %.3f sec/batch' % np.average(speeds))
        if len(speeds) > 1:
            print('First iteration: %.3f sec/batch' % speeds[0])
            print('Average excluding first iteration: %.3f sec/batch' %
                  np.average(speeds[1:]))

    return losses


class MnistConvBase(unittest.TestCase):
    def _runner(self):
        return None

    def _config(self, **kwargs):
        c = tf.ConfigProto()
        c.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
        c.allow_soft_placement = True
        return c

    @parameterized.expand([(25,), (50,), (100,)])
    def test_gpu(self, batch_size):
        def func():
            sess = tf.get_default_session()
            return self._runner()(sess, batch_size=batch_size)

        config = self._config(batch_size=batch_size)
        config.allow_soft_placement = True
        run_on_devices(func, '/device:GPU:0', config=config)

    @unittest.skip("No need to run on CPU")
    def test_cpu(self):
        def func():
            sess = tf.get_default_session()
            return self._runner()(sess)

        run_on_devices(func, '/device:CPU:0', config=self._config())

    @parameterized.expand([(25,), (50,), (100,)])
    def test_rpc(self, batch_size):
        def func():
            sess = tf.get_default_session()
            return self._runner()(sess, batch_size=batch_size)
        run_on_sessions(func, 'zrpc://tcp://127.0.0.1:5501', dev='/device:GPU:0',
                        config=self._config(batch_size=batch_size))

    def test_correctness(self):
        def func():
            sess = tf.get_default_session()
            return self._runner()(sess)

        actual, expected = run_on_rpc_and_gpu(func, config=self._config())
        assertAllClose(actual, expected, rtol=1e-3)


class TestMnistSoftmax(MnistConvBase):
    def _runner(self):
        return run_mnist_softmax


class TestMnistConv(MnistConvBase):
    def _runner(self):
        return run_mnist_conv

    def _config(self, **kwargs):
        MB = 1024 * 1024
        memusages = {
            25: (51.7 * MB - 38 * MB, 38 * MB),
            50: (64.8 * MB - 38 * MB, 38 * MB),
            100: (89 * MB - 38 * MB, 38 * MB),
        }
        batch_size = kwargs.get('batch_size', 50)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.salus_options.resource_map.temporary['MEMORY:GPU'] = memusages[batch_size][0]
        config.salus_options.resource_map.persistant['MEMORY:GPU'] = memusages[batch_size][1]
        return config


class TestMnistLarge(MnistConvBase):
    def _runner(self):
        return run_mnist_large

    def _config(self, **kwargs):
        MB = 1024 * 1024
        memusages = {
            25: (39 * MB - 23.5 * MB, 23.5 * MB),
            50: (54 * MB - 23.5 * MB, 23.5 * MB),
            100: (72 * MB - 23.5 * MB, 26 * MB),
        }
        batch_size = kwargs.get('batch_size', 50)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.salus_options.resource_map.temporary['MEMORY:GPU'] = memusages[batch_size][0]
        config.salus_options.resource_map.persistant['MEMORY:GPU'] = memusages[batch_size][1]
        return config


del MnistConvBase


if __name__ == '__main__':
    unittest.main()
