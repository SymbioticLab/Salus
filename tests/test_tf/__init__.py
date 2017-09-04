from contextlib import contextmanager

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from .lib import gradients  # NOQA: F401
from .lib import networks  # NOQA: F401
from .lib import datasets  # NOQA: F401


def _add_config_to_kwargs(kwargs, nconfig):
    if 'config' not in kwargs:
        kwargs['config'] = tf.ConfigProto()
    kwargs['config'].MergeFrom(nconfig)
    return kwargs


# TODO: implement the device selection as a session option.
# So we can test both CPU and GPU using the same code.
@contextmanager
def sess_and_device(target='', dev='', config=None, seed=None):
    finalCfg = tf.ConfigProto()
    finalCfg.graph_options.optimizer_options.do_constant_folding = False
    finalCfg.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
    finalCfg.device_count['RPC'] = 0
    if config is not None:
        finalCfg.MergeFrom(config)

    if seed is None:
        seed = 233

    with tf.Graph().as_default():
        tf.set_random_seed(seed)
        np.random.seed(seed)
        with tf.device(dev):
            with tf.Session(target, config=finalCfg) as sess:
                yield sess


def run_on_devices(func, devices, *args, **kwargs):
    if not isinstance(devices, (list, tuple)):
        devices = [devices] + list(args)

    results = []
    for d in devices:
        with sess_and_device(dev=d, **kwargs):
            res = func()
            results.append(res)
    return tuple(results)


def run_on_sessions(func, targets, *args, **kwargs):
    if not isinstance(targets, (list, tuple)):
        targets = [targets] + list(args)

    results = []
    for t in targets:
        with sess_and_device(t, **kwargs):
            res = func()
            results.append(res)
    return tuple(results)


def run_on_rpc_and_cpu(func, **kwargs):
    config = tf.ConfigProto()
    config.zmq_options.sched_cpu_only = True
    kwargs = _add_config_to_kwargs(kwargs, config)
    return run_on_sessions(func, 'zrpc://tcp://localhost:5501', '', **kwargs)


def run_on_rpc_and_gpu(func, **kwargs):
    config = tf.ConfigProto()
    config.zmq_options.sched_cpu_only = False
    config.allow_soft_placement = True
    kwargs = _add_config_to_kwargs(kwargs, config)
    return run_on_sessions(func, 'zrpc://tcp://localhost:5501', '', **kwargs)


def assertAllClose(actual, expected, **kwargs):
    def _assertAllClose(actual, expected, path):
        if isinstance(actual, (list, tuple)):
            for i, (a, e) in enumerate(zip(actual, expected)):
                _assertAllClose(a, e, path + [i])
        else:
            msg = "At element actual"
            if len(path) > 0:
                msg += '[{}]'.format(']['.join(str(i) for i in path))
            npt.assert_allclose(actual, expected, err_msg=msg, **kwargs)

    return _assertAllClose(actual, expected, [])
