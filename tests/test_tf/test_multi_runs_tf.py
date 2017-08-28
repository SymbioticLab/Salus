from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np

from parameterized import parameterized

from . import run_on_rpc_and_cpu, assertAllClose


class TestMultiRuns(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_multiruns(self):
        def func():
            a = tf.Variable(tf.zeros([2]))
            sess = tf.get_default_session()
            sess.run(tf.global_variables_initializer())
            return a.eval()

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

        def func2():
            a = tf.Variable(tf.zeros([1]))
            sess = tf.get_default_session()
            sess.run(tf.global_variables_initializer())
            return a.eval()

        actual, expected = run_on_rpc_and_cpu(func2)
        assertAllClose(actual, expected)


if __name__ == '__main__':
    unittest.main()
