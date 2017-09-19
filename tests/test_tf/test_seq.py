from __future__ import print_function

import unittest
from datetime import datetime
from timeit import default_timer

import tensorflow as tf

from . import run_on_rpc_and_cpu, run_on_rpc_and_gpu, run_on_sessions, run_on_devices
from . import networks, datasets
from .lib import tfhelper

def run_seq():
    pass


class SeqCaseBase(unittest.TestCase):
    def _seq(self):
        return None

    def test_gpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)

        run_on_devices(func, '/device:GPU:0', config=tf.ConfigProto(allow_soft_placement=True))

    def test_cpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)

        run_on_devices(func, '/device:CPU:0')

    def test_rpc_only(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)

        run_on_sessions(func, 'zrpc://tcp://127.0.0.1:5501')

    def test_fake_data(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)

        actual, expected = run_on_rpc_and_cpu(func, config=tf.ConfigProto(allow_soft_placement=True))
        self.assertEquals(actual, expected)

    @unittest.skip("Not yet implemented")
    def test_fake_data_gpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)

        actual, expected = run_on_rpc_and_gpu(func)
        self.assertEquals(actual, expected)

    def test_flowers(self):
        def func():
            def input_data(*a, **kw):
                return datasets.flowers_data(*a, height=224, width=224, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)

        actual, expected = run_on_rpc_and_cpu(func)
        self.assertEquals(actual, expected)

    @unittest.skip("Skip distributed runtime")
    def test_dist(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)

        run_on_sessions(func, 'grpc://localhost:2222')

    @unittest.skip("Skip distributed runtime")
    def test_dist_gpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_seq(self._vgg(), sess, input_data)
        run_on_devices(func, '/job:local/task:0/device:GPU:0', target='grpc://localhost:2222',
                       config=tf.ConfigProto(allow_soft_placement=True))


del SeqCaseBase


if __name__ == '__main__':
    unittest.main()
