from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from parameterized import parameterized, param

from . import run_on_rpc_and_cpu, device_and_sess

class TestBasicOps(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_variable(self):
        def func():
            a = tf.Variable(tf.zeros([200]))
            sess = tf.get_default_session()
            sess.run(tf.global_variables_initializer())
            return a.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

    def test_randomop(self):
        def func():
            seed = 233
            r = tf.random_normal([20, 20], seed=seed)
            return r.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_allclose(actual, expected, rtol=1e-6)

    def test_noop(self):
        with device_and_sess('/device:RPC:0') as sess:
            a = tf.no_op(name='mynoop')
            sess.run(a)

    @parameterized.expand([(tf.int8,), (tf.int16,), (tf.int32,), (tf.int64,)])
    def test_multiply_int(self, dtype):
        def func():
            a = tf.constant([3, 7], name='const_1', dtype=dtype)
            b = tf.constant([7, 3] , name='const_2', dtype=dtype)
            c = tf.constant(2, name='const_3', dtype=dtype)
            d = tf.multiply(a, b, name='mul_first')
            mul = tf.multiply(c, d, name='mul_second')
            return mul.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

    @parameterized.expand([(tf.float32,), (tf.float64,), (tf.complex64,), (tf.complex128,)])
    def test_multiply(self, dtype):
        def func():
            a = tf.constant([3, 7], name='const_1', dtype=dtype)
            b = tf.constant([7, 3] , name='const_2', dtype=dtype)
            c = tf.constant(2, name='const_3', dtype=dtype)
            d = tf.multiply(a, b, name='mul_first')
            mul = tf.multiply(c, d, name='mul_second')
            return mul.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

    @parameterized.expand([(tf.float32,), (tf.float64,), (tf.complex64,), (tf.complex128,)])
    def test_add(self, dtype):
        def func():
            a = tf.constant([3, 7], name='const_1', dtype=dtype)
            b = tf.constant([7, 3] , name='const_2', dtype=dtype)
            c = tf.constant(2, name='const_3', dtype=dtype)
            d = tf.add(a, b, name='add_first')
            add = tf.add(c, d, name='add_second')
            return add.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

    @parameterized.expand([(tf.int8,), (tf.int16,), (tf.int32,), (tf.int64,)])
    def test_add_int(self, dtype):
        def func():
            a = tf.constant([3, 7], name='const_1', dtype=dtype)
            b = tf.constant([7, 3] , name='const_2', dtype=dtype)
            c = tf.constant(2, name='const_3', dtype=dtype)
            d = tf.add(a, b, name='add_first')
            add = tf.add(c, d, name='add_second')
            return add.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

    def test_matmul(self):
        def func():
            m1 = np.random.normal(size=(20, 50))
            m2 = np.random.normal(size=(50, 60))
            m = tf.matmul(m1, m2)
            return m.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

    def test_conv2d(self):
        def func():
            image = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape=(1, 4, 3, 1), dtype=tf.float32)
            filter = tf.constant([1, 4, 7, 2, 5, 8, 3, 6, 9], shape=(3, 3, 1, 1), dtype=tf.float32)
            conv = tf.nn.conv2d(image, filter, [1, 1, 1, 1], 'SAME')
            return conv.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

    @parameterized.expand([(tf.float16,), (tf.float32,), (tf.float64,), (tf.int32,), (tf.int64,)])
    def test_relu(self, dtype):
        def func():
            np_features = np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(dtype.as_numpy_dtype)
            relu = tf.nn.relu(np_features)
            return relu.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        npt.assert_array_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()
