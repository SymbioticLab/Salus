from __future__ import print_function, absolute_import, division

import unittest

import tensorflow as tf
import numpy as np
from datetime import datetime
from timeit import default_timer

from parameterized import parameterized

from . import run_on_rpc_and_gpu, run_on_sessions, run_on_devices, assertAllClose
from . import networks
from .lib import tfhelper
from .lib.datasets import fake_data_ex


def run_superres(sess, input_data, batch_size=100):
    input_images, target_images = input_data(batch_size=batch_size)

    model = networks.SuperRes(input_images, target_images, batch_size=batch_size)
    model.build_model()

    salus_marker = tf.no_op(name="salus_main_iter")
    losses = []
    with tfhelper.initialized_scope(sess) as coord:
        speeds = []
        JCT = default_timer()
        for i in range(tfhelper.iteration_num_from_env()):
            if coord.should_stop():
                break
            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            _, loss_value, _ = sess.run([model.g_optim, model.g_loss, salus_marker])
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


class TestSuperRes(unittest.TestCase):
    def _config(self, **kwargs):
        KB = 1024
        MB = 1024 * KB
        memusages = {
            32: (252.79296875 * MB, 17.503280639648438 * MB),
            # 64: (496.98071670532227 * MB, 31.690780639648438 * MB),
            64: (500 * MB, 31.690780639648438 * MB),
            128: (992.9807167053223 * MB, 60.44078063964844 * MB),
        }
        batch_size = kwargs.get('batch_size', 100)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.salus_options.resource_map.temporary['MEMORY:GPU'] = memusages[batch_size][0]
        config.salus_options.resource_map.persistant['MEMORY:GPU'] = memusages[batch_size][1]
        return config

    def _get_func(self, batch_size):
        def func():
            def input_data(batch_size):
                variable_specs = [
                    ([32, 32, 3], {'dtype': tf.float32}, 'images'),
                    ([128, 128, 3], {'dtype': tf.float32}, 'targets'),
                ]
                input_images, target_images = fake_data_ex(batch_size, variable_specs=variable_specs)
                return input_images, target_images
            sess = tf.get_default_session()
            return run_superres(sess, input_data, batch_size=batch_size)
        return func

    @parameterized.expand([(32,), (64,), (128,)])
    def test_gpu(self, batch_size):
        config = self._config(batch_size=batch_size)
        config.allow_soft_placement = True
        run_on_devices(self._get_func(batch_size), '/device:GPU:0', config=config)

    @parameterized.expand([(32,), (64,), (128,)])
    def test_distributed(self, batch_size):
        run_on_sessions(self._get_func(batch_size),
                        'grpc://localhost:2345',
                        dev='/job:tfworker/device:GPU:0',
                        config=self._config(batch_size=batch_size))

    @parameterized.expand([(64,)])
    @unittest.skip("No need to run on CPU")
    def test_cpu(self, batch_size):
        run_on_devices(self._get_func(batch_size), '/device:CPU:0',
                       config=self._config(batch_size=batch_size))

    @parameterized.expand([(32,), (64,), (128,)])
    def test_rpc(self, batch_size):
        run_on_sessions(self._get_func(batch_size),
                        'zrpc://tcp://127.0.0.1:5501', dev='/device:GPU:0',
                        config=self._config(batch_size=batch_size))

    @parameterized.expand([(64,)])
    def test_correctness(self, batch_size):
        config = self._config(batch_size=batch_size)
        config.allow_soft_placement = True
        actual, expected = run_on_rpc_and_gpu(self._get_func(batch_size), config=config)
        assertAllClose(actual, expected)


if __name__ == '__main__':
    unittest.main()
