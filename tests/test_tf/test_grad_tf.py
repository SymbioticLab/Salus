from __future__ import print_function

import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from parameterized import parameterized

from . import run_on_rpc_and_cpu, assertAllClose
from .gradients import compute_gradient


class TestOpGradients(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    @parameterized.expand([(tf.int8,), (tf.int16,), (tf.int32,), (tf.int64,)])
    def test_multiply_int(self, dtype):
        def func():
            x_init = np.asarray(
                [[-9, -7, -5, -3, -1], [1, 3, 5, 7, 9]],
                dtype=dtype.as_numpy_dtype, order="F")
            x = tf.constant(
                [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9],
                shape=[2, 5], dtype=dtype,
                name="x")
            y = tf.constant(
                [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9],
                shape=[2, 5], dtype=dtype,
                name="y")
            z = tf.multiply(x, y, name="mul_test")
            return compute_gradient(x, [2, 5], z, [2, 5], x_init_value=x_init)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    @parameterized.expand([(tf.float32,), (tf.float64,), (tf.complex64,), (tf.complex128,)])
    def test_multiply(self, dtype):
        def func():
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
                dtype=dtype.as_numpy_dtype, order="F")
            return compute_gradient(x, [2, 5], z, [2, 5], x_init_value=x_init)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    @parameterized.expand([(tf.int8,), (tf.int16,), (tf.int32,), (tf.int64,)])
    def test_add_int(self, dtype):
        def func():
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
                dtype=dtype.as_numpy_dtype, order="F")
            return compute_gradient(
                x, [2, 5], z, [2, 5], x_init_value=x_init)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    @parameterized.expand([(tf.float32,), (tf.float64,), (tf.complex64,), (tf.complex128,)])
    def test_add(self, dtype):
        def func():
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
                dtype=dtype.as_numpy_dtype, order="F")
            return compute_gradient(x, [2, 5], z, [2, 5], x_init_value=x_init)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    @parameterized.expand([(tf.float32,), (tf.float64,), (tf.complex64,), (tf.complex128,)])
    def test_matmul(self, dtype):
        def func():
            m1 = np.random.normal(size=(2, 5))
            m2 = np.random.normal(size=(5, 6))
            m3 = np.matmul(m1, m2)
            x = tf.constant(m1, dtype=dtype, name="x")
            y = tf.constant(m2, dtype=dtype, name="y")
            z = tf.matmul(x, y, name="matmul_test")
            dx = compute_gradient(x, m1.shape, z, m3.shape, x_init_value=m1)
            dy = compute_gradient(y, m2.shape, z, m3.shape, x_init_value=m2)
            return dx, dy

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    @parameterized.expand([(tf.float16,), (tf.float32,)])
    def test_conv2d(self, dtype):
        def func():
            mi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                          dtype=dtype.as_numpy_dtype).reshape((1, 4, 3, 1))
            mf = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9],
                          dtype=dtype.as_numpy_dtype).reshape((3, 3, 1, 1))
            image = tf.constant(mi, dtype=dtype, name="image")
            filter = tf.constant(mf, dtype=dtype, name="filter")
            z = tf.nn.conv2d(image, filter, [1, 1, 1, 1], 'SAME')

            di = compute_gradient(image, mi.shape, z, mi.shape, x_init_value=mi)
            df = compute_gradient(filter, mf.shape, z, mi.shape, x_init_value=mf)
            return di, df

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    @parameterized.expand([(tf.float32,), (tf.float64,)])
    def test_grad_relu(self, dtype):
        def func():
            x = tf.constant([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                            shape=[2, 5], dtype=dtype, name="x")
            y = tf.nn.relu(x, name="relu")
            x_init = np.asarray([[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                                dtype=dtype.as_numpy_dtype, order="F")
            return compute_gradient(x, [2, 5], y, [2, 5], x_init_value=x_init)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    @parameterized.expand([(tf.float32,), (tf.float64,)])
    def test_grad_grad_relu(self, dtype):
        def func():
            x = tf.constant([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                            shape=[2, 5], dtype=dtype, name="x")
            y = tf.nn.relu(x, name="relu")
            z = tf.gradients(y, x)
            x_init = np.asarray([[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                                dtype=dtype.as_numpy_dtype, order="F")
            return compute_gradient(x, [2, 5], z[0], [2, 5], x_init_value=x_init)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    def test_grad_relu_scala(self):
        def func():
            x = tf.Variable(100.)
            y = tf.nn.relu(x)
            loss = y**2
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.25)
            train_op = optimizer.minimize(loss)

            sess = tf.get_default_session()
            sess.run(tf.global_variables_initializer())
            sess.run(train_op)
            return x.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)


if __name__ == '__main__':
    unittest.main()
