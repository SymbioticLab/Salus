from __future__ import print_function

import unittest

import tensorflow as tf

from parameterized import parameterized

from . import run_on_rpc_and_cpu, run_on_sessions, run_on_devices
from . import networks, datasets
from .lib import tfhelper
from .lib.seq2seq.ptb.ptb_word_lm import get_config


def run_seq_ptb(sess, config_name):
    config = get_config(config_name)
    eval_config = get_config(config_name)
    config.max_max_epoch = 1
    eval_config.max_max_epoch = 1
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    train_input, valid_input, test_input = datasets.ptb_data(config, eval_config)
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = networks.PTBModel(is_training=True, config=config, input_=train_input)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = networks.PTBModel(is_training=False, config=eval_config, input_=valid_input)
        tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = networks.PTBModel(is_training=False, config=eval_config, input_=test_input)

    with tfhelper.initialized_scope(sess) as coord:
        for i in range(config.max_max_epoch):
            if coord.should_stop():
                break

            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(sess, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
            train_perplexity = m.run_epoch(sess, eval_op=m.train_op, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = mvalid.run_epoch(sess)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = mtest.run_epoch(sess)
        print("Test Perplexity: %.3f" % test_perplexity)


configs = ['tiny', 'small', 'medium', 'large']


class SeqCaseBase(unittest.TestCase):
    def _runner(self):
        return None

    def get_func_to_run(self, config_name):
        return lambda: self._runner()(tf.get_default_session(), config_name)

    @parameterized.expand(configs)
    def test_gpu(self, config_name):
        run_on_devices(self.get_func_to_run(config_name), '/device:GPU:0',
                       config=tf.ConfigProto(allow_soft_placement=True))

    @parameterized.expand(configs)
    def test_cpu(self, config_name):
        run_on_devices(self.get_func_to_run(config_name), '/device:CPU:0')

    @parameterized.expand(configs)
    def test_rpc(self, config_name):
        run_on_sessions(self.get_func_to_run(config_name), 'zrpc://tcp://127.0.0.1:5501')

    @parameterized.expand(configs)
    def test_correctness(self, config_name):
        actual, expected = run_on_rpc_and_cpu(self.get_func_to_run(config_name))
        self.assertEquals(actual, expected)


class TestSeqPtb(SeqCaseBase):
    def _runner(self):
        return run_seq_ptb


del SeqCaseBase


if __name__ == '__main__':
    unittest.main()
