from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

class TestBasicOps(unittest.TestCase):

    def test_variable(self):
        with tf.device('/device:RPC:0'):
            a = tf.Variable(tf.zeros([200]))
        with tf.device('/device:CPU:0'):
            b = tf.Variable(tf.zeros([200]))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            npt.assert_array_equal(sess.run(a), sess.run(b))

    def test_randomop(self):
        with tf.Session() as sess:
            tf.set_random_seed(233)
            with tf.device('/device:RPC:0'):
                a = tf.random_normal([20, 20])
                actual = sess.run(a)

        with tf.Session() as sess:
            tf.set_random_seed(233)
            with tf.device('/device:CPU:0'):
                b = tf.random_normal([20, 20])
                expected = sess.run(b)

        npt.assert_array_equal(actual, expected)

    def test_noop(self):
        with tf.device('/device:RPC:0'):
            a = tf.no_op(name='mynoop')

        with tf.Session() as sess:
            try:
                sess.run(a)
            except:
                self.fail("running noop should not raise exception")

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

        with tf.Session() as sess:
            npt.assert_array_equal(sess.run(actual), sess.run(expected))

    def test_matmul(self):
        m1 = np.random.normal(size=(20, 20))
        m2 = np.random.normal(size=(20, 20))
        with tf.device('/device:RPC:0'):
            actual = tf.matmul(m1, m2)
        with tf.device('/device:CPU:0'):
            expected = tf.matmul(m1, m2)

        with tf.Session() as sess:
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

        with tf.Session() as sess:
            npt.assert_array_equal(sess.run(actual), sess.run(expected))


if __name__ == '__main__':
    unittest.main()
