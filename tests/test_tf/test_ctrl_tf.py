from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np

from parameterized import parameterized

from . import run_on_rpc_and_cpu, device_and_sess, assertAllClose


class TestBasicOps(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_switch(self):
        def func():
            pass

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)

    def test_merge(self):
        def func():
            pass

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)


if __name__ == '__main__':
    unittest.main()
