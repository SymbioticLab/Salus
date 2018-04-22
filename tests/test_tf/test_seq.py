from __future__ import print_function

import unittest
from typing import Callable

import numpy as np
import tensorflow as tf

from parameterized import parameterized

from . import run_on_rpc_and_gpu, run_on_sessions, run_on_devices
from . import networks, datasets
from .lib import tfhelper
from .lib.seq2seq.ptb.ptb_word_lm import get_config


def run_seq_ptb(sess, config_name):
    config = get_config(config_name)
    eval_config = get_config(config_name)
    config.max_max_epoch = 1
    config.max_epoch = 1

    eval_config.max_max_epoch = 1
    eval_config.max_epoch = 1
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    train_input, valid_input, test_input = datasets.ptb_data(config, eval_config)
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    # force to run only a few steps
    train_input.epoch_size = tfhelper.iteration_num_from_env()
    valid_input.epoch_size = train_input.epoch_size
    test_input.epoch_size = train_input.epoch_size

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = networks.PTBModel(is_training=True, config=config, input_=train_input)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

    with tfhelper.initialized_scope(sess) as coord:
        for i in range(config.max_max_epoch):
            if coord.should_stop():
                break

            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0)
            m.assign_lr(sess, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
            train_perplexity, speeds = m.run_epoch(sess, eval_op=m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            print('Average: %.3f sec/batch' % np.average(speeds))
            if len(speeds) > 1:
                print('First iteration: %.3f sec/batch' % speeds[0])
                print('Average excluding first iteration: %.3f sec/batch' % np.average(speeds[1:]))


model_sizes = ['tiny', 'small', 'medium', 'large']


class SeqCaseBase(unittest.TestCase):
    def _runner(self):
        # type: () -> Callable
        raise NotImplementedError

    def _config(self, model_size):
        raise NotImplementedError

    def get_func_to_run(self, config_name):
        return lambda: self._runner()(tf.get_default_session(), config_name)

    @parameterized.expand(model_sizes)
    def test_gpu(self, model_size):
        run_on_devices(self.get_func_to_run(model_size), '/device:GPU:0',
                       config=tf.ConfigProto(allow_soft_placement=True))

    @parameterized.expand(model_sizes)
    @unittest.skip("No need to run on CPU")
    def test_cpu(self, model_size):
        run_on_devices(self.get_func_to_run(model_size), '/device:CPU:0')

    @parameterized.expand(model_sizes)
    def test_rpc(self, model_size):
        run_on_sessions(self.get_func_to_run(model_size), 'zrpc://tcp://127.0.0.1:5501', dev='/device:GPU:0',
                        config=self._config(model_size))

    @parameterized.expand(['small'])
    def test_correctness(self, model_size):
        actual, expected = run_on_rpc_and_gpu(self.get_func_to_run(model_size),
                                              config=self._config(model_size))
        self.assertEquals(actual, expected)


# @unittest.skip("Too slow")
class TestSeqPtb(SeqCaseBase):
    def _runner(self):
        return run_seq_ptb

    def _config(self, model_size):
        KB = 1024
        MB = 1024 * KB
        GB = 1024 * MB
        memusages = {
            'tiny': (1.9 * MB, 200 * KB),
            'small': (79.2 * MB, 18 * MB),
            'medium': (1.01 * GB, 80 * MB),
            'large': (5.05 * GB, 343.3 * MB),
        }

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.salus_options.resource_map.temporary['MEMORY:GPU'] = memusages[model_size][0]
        config.salus_options.resource_map.persistant['MEMORY:GPU'] = memusages[model_size][1]
        return config


del SeqCaseBase


if __name__ == '__main__':
    unittest.main()
