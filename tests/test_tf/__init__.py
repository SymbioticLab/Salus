from contextlib import contextmanager

import numpy as np
import tensorflow as tf


@contextmanager
def device_and_sess(dev, config=None, seed=None):
    if config is None:
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.do_constant_folding = False
        config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0

    if seed is None:
        seed = 233

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        np.random.seed(seed)
        with tf.device(dev):
            with tf.Session(config=config):
                yield
