from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

class TestBasicOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = tf.ConfigProto()
        cls.config.graph_options.optimizer_options.do_constant_folding = False

    def setUp(self):
        tf.reset_default_graph()

    def test_variable(self):
        with tf.device('/device:RPC:0'):
            a = tf.Variable(tf.zeros([200]))
        with tf.device('/device:CPU:0'):
            b = tf.Variable(tf.zeros([200]))

        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            npt.assert_array_equal(sess.run(a), sess.run(b))

    def test_randomop(self):
        seed = 233

        with tf.device('/device:RPC:0'):
            actual = tf.random_normal([20, 20], seed=seed)
        with tf.device('/device:CPU:0'):
            expected = tf.random_normal([20, 20], seed=seed)

        with tf.Session(config=self.config) as sess:
            npt.assert_array_equal(sess.run(actual), sess.run(expected))

    def test_noop(self):
        with tf.device('/device:RPC:0'):
            a = tf.no_op(name='mynoop')
        with tf.Session(config=self.config) as sess:
            sess.run(a)

    def test_multiply(self):
        with tf.device('/device:RPC:0'):
            a = tf.constant([3, 7], name='const_1')
            b = tf.constant([7, 3] , name='const_2')
            c = tf.constant(2, name='const_3')
            d = tf.multiply(a, b, name='mul_first')
            actual = tf.multiply(c, d, name='mul_second')

        with tf.device('/device:CPU:0'):
            a = tf.constant([3, 7], name='const_1')
            b = tf.constant([7, 3] , name='const_2')
            c = tf.constant(2, name='const_3')
            d = tf.multiply(a, b, name='mul_first')
            expected = tf.multiply(c, d, name='mul_second')

        with tf.Session(config=self.config) as sess:
            npt.assert_array_equal(sess.run(actual), sess.run(expected))

    def test_matmul(self):
        m1 = np.random.normal(size=(20, 20))
        m2 = np.random.normal(size=(20, 20))
        with tf.device('/device:RPC:0'):
            actual = tf.matmul(m1, m2)
        with tf.device('/device:CPU:0'):
            expected = tf.matmul(m1, m2)

        with tf.Session(config=self.config) as sess:
            npt.assert_array_equal(sess.run(actual), sess.run(expected))

    def test_conv2d(self):
        with tf.device('/device:RPC:0'):
            image = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape=(1, 4, 3, 1), dtype=tf.float32)
            filter = tf.constant([1, 4, 7, 2, 5, 8, 3, 6, 9], shape=(3, 3, 1, 1), dtype=tf.float32)
            actual = tf.nn.conv2d(image, filter, [1, 1, 1, 1], 'SAME')
        with tf.device('/device:CPU:0'):
            image = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape=(1, 4, 3, 1), dtype=tf.float32)
            filter = tf.constant([1, 4, 7, 2, 5, 8, 3, 6, 9], shape=(3, 3, 1, 1), dtype=tf.float32)
            expected = tf.nn.conv2d(image, filter, [1, 1, 1, 1], 'SAME')

        with tf.Session(config=self.config) as sess:
            npt.assert_array_equal(sess.run(actual), sess.run(expected))


if __name__ == '__main__':
    unittest.main()
