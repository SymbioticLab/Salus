from contextlib import contextmanager

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from .lib import gradients  # NOQA: F401
from .lib import networks  # NOQA: F401
from .lib import datasets  # NOQA: F401


# TODO: implement the device selection as a session option.
# So we can test both CPU and GPU using the same code.
@contextmanager
def device_and_sess(dev, config=None, seed=None):
    finalCfg = tf.ConfigProto()
    finalCfg.graph_options.optimizer_options.do_constant_folding = False
    finalCfg.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
    if config is not None:
        finalCfg.MergeFrom(config)

    if seed is None:
        seed = 233

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        np.random.seed(seed)
        with tf.device(dev):
            with tf.Session(config=finalCfg) as sess:
                yield sess


def run_on_devices(func, devices, *args, **kwargs):
    if not isinstance(devices, (list, tuple)):
        devices = [devices] + list(args)

    results = []
    for d in devices:
        with device_and_sess(d, **kwargs):
            res = func()
            results.append(res)
    return tuple(results)


def run_on_rpc_and_cpu(func, **kwargs):
    return run_on_devices(func, '/device:RPC:0', '/device:CPU:0', **kwargs)


def assertAllClose(actual, expected):
    def _assertAllClose(actual, expected, path):
        if isinstance(actual, (list, tuple)):
            for i, (a, e) in enumerate(zip(actual, expected)):
                _assertAllClose(a, e, path + [i])
        else:
            msg = "At element actual"
            if len(path) > 0:
                msg += '[{}]'.format(']['.join(str(i) for i in path))
            npt.assert_allclose(actual, expected, err_msg=msg)

    return _assertAllClose(actual, expected, [])
