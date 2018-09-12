from __future__ import print_function, absolute_import, division

import unittest

import numpy as np
import tensorflow as tf
from datetime import datetime
from timeit import default_timer

from parameterized import parameterized

from . import run_on_rpc_and_gpu, run_on_sessions, run_on_devices, assertAllClose, tfDistributedEndpointOrSkip
from . import networks
from .lib import tfhelper
from .lib.datasets import fake_data


IMAGE_SIZE_MNIST = 28


def run_vae(sess, args=None, isEval=False):
    """ parameters """
    if args is None:
        args = networks.vae.get_args()

    dim_img = IMAGE_SIZE_MNIST ** 2  # number of pixels for a MNIST image
    x_image, _, num_classes = fake_data(args.batch_size, None, height=IMAGE_SIZE_MNIST, width=IMAGE_SIZE_MNIST,
                                        depth=1, num_classes=10)

    with tf.name_scope('model'):
        # inputs
        x = tf.reshape(x_image, [-1, 784], name='target_img')
        x_hat = tf.identity(x, name='input_img')

        # dropout
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # network architecture
        y, z, loss, neg_marginal_likelihood, kl_divergence = networks.vae.autoencoder(x_hat, x, dim_img, args.dim_z,
                                                                                      args.n_hidden, keep_prob)
        batch_step = loss

        if not isEval:
            # optimization
            train_op = tf.train.AdamOptimizer(args.learn_rate).minimize(loss)
            batch_step = train_op

    """ training """
    salus_marker = tf.no_op(name="salus_main_iter")
    inbetween = []
    last_end_time = 0
    losses = []
    with tfhelper.initialized_scope(sess) as coord:
        speeds = []
        JCT = default_timer()
        # Loop over all batches
        for i in range(tfhelper.iteration_num_from_env()):
            if coord.should_stop():
                break

            print("{}: Start running step {}".format(datetime.now(), i))
            start_time = default_timer()
            _, _, loss_value = sess.run([batch_step, salus_marker, loss], feed_dict={keep_prob: 1 if isEval else 0.9})
            end_time = default_timer()

            if last_end_time > 0:
                inbetween.append(start_time - last_end_time)
            last_end_time = end_time

            duration = end_time - start_time
            examples_per_sec = args.batch_size / duration
            sec_per_batch = float(duration)
            speeds.append(sec_per_batch)

            losses.append(loss_value)
            fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
            print(fmt_str.format(datetime.now(), i,
                                 loss_value, examples_per_sec, sec_per_batch))

        JCT = default_timer() - JCT
        print('Training time is %.3f sec' % JCT)
        print('Average: %.3f sec/batch' % np.average(speeds))
        if len(speeds) > 1:
            print('First iteration: %.3f sec/batch' % speeds[0])
            print('Average excluding first iteration: %.3f sec/batch' % np.average(speeds[1:]))

    return losses


class TestVae(unittest.TestCase):
    def _config(self, args, isEval=False):
        KB = 1024
        MB = 1024 * KB
        if isEval:
            memusages = {
                1: (2 * MB, 5.1 * MB),
                5: (2 * MB, 5.1 * MB),
                10: (2 * MB, 5.1 * MB),
            }
        else:
            memusages = {
                64: (5 * MB, 21 * MB),
                128: (7 * MB, 22 * MB),
                256: (15 * MB, 23 * MB),
            }

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.salus_options.resource_map.temporary['MEMORY:GPU'] = memusages[args.batch_size][0]
        config.salus_options.resource_map.persistant['MEMORY:GPU'] = memusages[args.batch_size][1]
        return config

    @parameterized.expand([(1,), (5,), (10,)])
    def test_gpu_eval(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_devices(lambda: run_vae(tf.get_default_session(), args, True), '/device:GPU:0',
                       config=self._config(args, True))

    @parameterized.expand([(64,), (128,), (256,)])
    def test_gpu(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_devices(lambda: run_vae(tf.get_default_session(), args), '/device:GPU:0',
                       config=self._config(args))

    @parameterized.expand([(1,), (5,), (10,)])
    def test_distributed_eval(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_sessions(lambda: run_vae(tf.get_default_session(), args, True),
                        tfDistributedEndpointOrSkip(),
                        dev='/job:tfworker/device:GPU:0',
                        config=self._config(args, True))

    @parameterized.expand([(64,), (128,), (256,)])
    def test_distributed(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_sessions(lambda: run_vae(tf.get_default_session(), args),
                        tfDistributedEndpointOrSkip(),
                        dev='/job:tfworker/device:GPU:0',
                        config=self._config(args))

    @parameterized.expand([(1,), (5,), (10,)])
    @unittest.skip("No need to run on CPU")
    def test_cpu_eval(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_devices(lambda: run_vae(tf.get_default_session(), args, True), '/device:CPU:0',
                       config=self._config(args, True))

    @parameterized.expand([(64,), (128,), (256,)])
    @unittest.skip("No need to run on CPU")
    def test_cpu(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_devices(lambda: run_vae(tf.get_default_session(), args), '/device:CPU:0',
                       config=self._config(args))

    @parameterized.expand([(1,), (5,), (10,)])
    def test_rpc_eval(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_sessions(lambda: run_vae(tf.get_default_session(), args, True),
                        'zrpc://tcp://127.0.0.1:5501', dev='/device:GPU:0',
                        config=self._config(args, True))

    @parameterized.expand([(64,), (128,), (256,)])
    def test_rpc(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        run_on_sessions(lambda: run_vae(tf.get_default_session(), args),
                        'zrpc://tcp://127.0.0.1:5501', dev='/device:GPU:0',
                        config=self._config(args))

    @parameterized.expand([(64,), (128,), (256,)])
    def test_correctness(self, batch_size):
        args = networks.vae.get_args(batch_size=batch_size)
        actual, expected = run_on_rpc_and_gpu(lambda: run_vae(tf.get_default_session(), args),
                                              config=self._config(args))
        assertAllClose(actual, expected)


if __name__ == '__main__':
    unittest.main()
