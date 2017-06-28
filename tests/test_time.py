import tensorflow as tf
from timeit import default_timer, timeit
import sys

config = tf.ConfigProto()
config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
config.graph_options.optimizer_options.do_constant_folding = False

def build_graph(x):
    m = tf.matmul(tf.random_normal((x,x)), tf.random_normal((x,x)))
    return tf.reduce_sum(m)

def time_on(dev, x, rep = 10):
    print('-----------------------------------------------------------------')
    print('Run on {} with {} repetations, data size {}'.format(dev, rep, x))
    tf.reset_default_graph()
    with tf.device('/device:' + dev + ':0'):
        op = build_graph(x)
        with tf.Session(config=config) as sess:
            sess.run(op)  # warm up
            st = default_timer()
            for _ in range(rep):
                sess.run(op)
            dur = (default_timer() - st) / rep
            print('Average time per run: {:.5f}s'.format(dur))
    print('=================================================================')

def gpu(x, rep):
    return time_on('GPU', x, rep)

def cpu(x, rep):
    return time_on('CPU', x, rep)

def rpc(x, rep):
    return time_on('RPC', x, rep)

argc = len(sys.argv)
size = 200
rep = 10

if argc > 1:
    size = int(sys.argv[1])
if argc > 2:
    rep = int(sys.argv[2])

gpu(size, rep)
cpu(size, rep)
rpc(size, rep)
