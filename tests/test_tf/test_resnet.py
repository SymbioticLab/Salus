from __future__ import print_function

import unittest
from datetime import datetime
from timeit import default_timer

import numpy as np
import tensorflow as tf

from parameterized import parameterized

from . import run_on_rpc_and_cpu, run_on_sessions, run_on_devices
from . import networks, datasets
from .lib import tfhelper


def run_resnet(sess, input_data, batch_size=100):
    images, labels, num_classes = input_data(batch_size=batch_size)

    hps = networks.ResNetHParams(batch_size=batch_size,
                                 num_classes=num_classes,
                                 min_lrn_rate=0.0001,
                                 lrn_rate=0.1,
                                 num_residual_units=5,
                                 use_bottleneck=False,
                                 weight_decay_rate=0.0002,
                                 relu_leakiness=0.1,
                                 optimizer='mom')
    resnet = networks.ResNet(hps, images, labels, "train")

    resnet.build_graph()

    losses = []
    with tfhelper.initialized_scope(sess) as coord:
        speeds = []
        JCT = default_timer()
        for i in range(tfhelper.iteration_num_from_env()):
            if coord.should_stop():
                break
            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            _, loss_value = sess.run([resnet.train_op, resnet.cost])
            end_time = default_timer()

            duration = end_time - start_time
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            speeds.append(sec_per_batch)
            fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
            print(fmt_str.format(datetime.now(), i, loss_value, examples_per_sec, sec_per_batch))
            losses.append(loss_value)

        JCT = default_timer() - JCT
        print('Training time is %.3f sec' % JCT)
        print('Average: %.3f sec/batch' % np.average(speeds))
        if len(speeds) > 1:
            print('First iteration: %.3f sec/batch' % speeds[0])
            print('Average excluding first iteration: %.3f sec/batch' % np.average(speeds[1:]))

    return losses


class ResNetCaseBase(unittest.TestCase):
    def _config(self, **kwargs):
        return tf.ConfigProto()

    def _get_func(self, batch_size):
        return None

    @parameterized.expand([(25,), (50,), (75,)])
    def test_gpu(self, batch_size):
        config = self._config(batch_size=batch_size)
        config.allow_soft_placement = True
        run_on_devices(self._get_func(batch_size), '/device:GPU:0', config=config)

    @parameterized.expand([(25,), (50,), (75,)])
    @unittest.skip("No need to run on CPU")
    def test_cpu(self, batch_size):
        run_on_devices(self._get_func(batch_size), '/device:CPU:0',
                       config=self._config(batch_size=batch_size))

    @parameterized.expand([(25,), (50,), (75,)])
    def test_rpc(self, batch_size):
        run_on_sessions(self._get_func(batch_size),
                        'zrpc://tcp://127.0.0.1:5501',
                        config=self._config(batch_size=batch_size))

    @parameterized.expand([(25,), (50,), (75,)])
    def test_correctness(self, batch_size):
        config = self._config(batch_size=batch_size)
        config.allow_soft_placement = True
        actual, expected = run_on_rpc_and_cpu(self._get_func(batch_size), config=config)
        self.assertEquals(actual, expected)


@unittest.skip("Fake data is used as common dataset instead")
class TestResNetCifar10(ResNetCaseBase):
    def _config(self, **kwargs):
        # FIXME: update memory usage
        memusages = {
            25: (6935520748 - 1661494764, 1661494764),
            50: (10211120620 - 1662531248, 1662531248),
            75: (10211120620 - 1662531248, 1662531248),
            100: (11494955340, 1.67e9),
        }
        batch_size = kwargs.get('batch_size', 100)

        config = tf.ConfigProto()
        config.zmq_options.resource_map.temporary['MEMORY:GPU'] = memusages[batch_size][0]
        config.zmq_options.resource_map.persistant['MEMORY:GPU'] = memusages[batch_size][1]
        return config

    def _get_func(self, batch_size):
        def func():
            def input_data(*a, **kw):
                return datasets.cifar10_data(*a, **kw)
            sess = tf.get_default_session()
            return run_resnet(sess, input_data, batch_size=batch_size)
        return func


class TestResNetFakeData(ResNetCaseBase):
    def _config(self, **kwargs):
        MB = 1024 * 1024
        memusages = {
            25: (5123864571 - 38.1 * MB, 38.1 * MB),
            50: (10063719070 - 38.1 * MB, 8 * MB),
            75: (14 * 1024 * MB - 8 * MB, 8 * MB),
        }
        batch_size = kwargs.get('batch_size', 100)

        config = tf.ConfigProto()
        config.zmq_options.resource_map.temporary['MEMORY:GPU'] = memusages[batch_size][0]
        config.zmq_options.resource_map.persistant['MEMORY:GPU'] = memusages[batch_size][1]
        return config

    def _get_func(self, batch_size):
        def func():
            def input_data(*a, **kw):
                images, labels, num_classes = datasets.fake_data(*a, batch_num=50, height=224,
                                                                 width=224, num_classes=1000, **kw)
                labels = tf.one_hot(labels, num_classes, axis=-1)
                return images, labels, num_classes
            sess = tf.get_default_session()
            return run_resnet(sess, input_data, batch_size=batch_size)
        return func


del ResNetCaseBase


if __name__ == '__main__':
    unittest.main()
