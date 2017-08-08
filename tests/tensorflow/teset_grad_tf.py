from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt
from ddt import ddt, data, unpack

@ddt
class TestOpGradients(unittest.TestCase):

    def get_session(self):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.do_constant_folding = False
        config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
        return tf.Session(config=config)

    def setUp(self):
        tf.reset_default_graph()

    # TODO: implement the device selection as a session option.
    # So we can test both CPU and GPU using the same code.
    @data(tf.int8, tf.int16, tf.int32, tf.int64)
    def test_multiply_int(self, dtype):
        with self.get_session():
            with tf.device('/device:RPC:0'):
                x = tf.constant(
                    [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9],
                    shape=[2, 5], dtype=dtype,
                    name="x")
                y = tf.constant(
                    [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9],
                    shape=[2, 5], dtype=dtype,
                    name="y")
                z = tf.multiply(x, y, name="mul_test")
                x_init = np.asarray(
                    [[-9, -7, -5, -3, -1], [1, 3, 5, 7, 9]],
                    dtype=dtype, order="F")
                err = tf.test.gradient_checker.compute_gradient_error(
                    x, [2, 5], z, [2, 5], x_init_value=x_init)
                self.assertLess(err, 1e-4)

    @data(tf.float32, tf.double, tf.complex64, tf.complex128)
    def test_multiply(self, dtype):
        with self.get_session():
            with tf.device('/device:RPC:0'):
                x = tf.constant(
                    [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                    shape=[2, 5], dtype=dtype,
                    name="x")
                y = tf.constant(
                    [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                    shape=[2, 5], dtype=dtype,
                    name="y")
                z = tf.multiply(x, y, name="mul_test")
                x_init = np.asarray(
                    [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                    dtype=dtype, order="F")
                err = tf.test.gradient_checker.compute_gradient_error(
                    x, [2, 5], z, [2, 5], x_init_value=x_init)
                self.assertLess(err, 1e-4)

    @data(tf.int8, tf.int16, tf.int32, tf.int64)
    def test_add_int(self, dtype):
        with self.get_session():
            with tf.device('/device:RPC:0'):
                x = tf.constant(
                    [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9],
                    shape=[2, 5], dtype=dtype,
                    name="x")
                y = tf.constant(
                    [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9],
                    shape=[2, 5], dtype=dtype,
                    name="y")
                z = tf.add(x, y, name="add_test")
                x_init = np.asarray(
                    [[-9, -7, -5, -3, -1], [1, 3, 5, 7, 9]],
                    dtype=dtype, order="F")
                err = tf.test.gradient_checker.compute_gradient_error(
                    x, [2, 5], z, [2, 5], x_init_value=x_init)
                self.assertLess(err, 1e-4)

    @data(tf.float32, tf.double, tf.complex64, tf.complex128)
    def test_add(self, dtype):
        with self.get_session():
            with tf.device('/device:RPC:0'):
                x = tf.constant(
                    [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                    shape=[2, 5], dtype=dtype,
                    name="x")
                y = tf.constant(
                    [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                    shape=[2, 5], dtype=dtype,
                    name="y")
                z = tf.add(x, y, name="add_test")
                x_init = np.asarray(
                    [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                    dtype=dtype, order="F")
                err = tf.test.gradient_checker.compute_gradient_error(
                    x, [2, 5], z, [2, 5], x_init_value=x_init)
                self.assertLess(err, 1e-4)

    @data(tf.int8, tf.int16, tf.int32, tf.int64,
          tf.float32, tf.double, tf.complex64, tf.complex128)
    def test_matmul(self, dtype):
        m1 = np.random.normal(size=(20, 50), dtype=dtype)
        m2 = np.random.normal(size=(50, 60), dtype=dtype)
        m3 = np.matmul(m1, m2)
        with self.get_session():
            with tf.device('/device:RPC:0'):
                x = tf.constant(m1, dtype=dtype, name="x")
                y = tf.constant(m2, dtype=dtype, name="y")
                z = tf.matmul(x, y, name="matmul_test")
                err = tf.test.gradient_checker.compute_gradient_error(
                    x, m1.shape, z, m3, x_init_value=m1)
                self.assertLess(err, 1e-4)
                err = tf.test.gradient_checker.compute_gradient_error(
                    y, m2.shape, z, m3, x_init_value=m2)
                self.assertLess(err, 1e-4)


    @data(tf.int8, tf.int16, tf.int32, tf.int64,
          tf.float32, tf.double, tf.complex64, tf.complex128)
    def test_conv2d(self, dtype):
        mi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape=(1, 4, 3, 1), dtype=dtype)
        mf = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9], shape=(3, 3, 1, 1), dtype=dtype)
        with self.get_session():
            with tf.device('/device:RPC:0'):
                image = tf.constant(mi, dtype=dtype, name="image")
                filter = tf.constant(mf, dtype=dtype, name="filter")
                z = tf.nn.conv2d(image, filter, [1, 1, 1, 1], 'SAME')

                err = tf.test.gradient_checker.compute_gradient_error(
                    image, mi.shape, z, mi.shape, x_init_value=mi)
                self.assertLess(err, 1e-4)
                err = tf.test.gradient_checker.compute_gradient_error(
                    filter, mf.shape, z, mi.shape, x_init_value=mf)
                self.assertLess(err, 1e-4)


if __name__ == '__main__':
    unittest.main()
