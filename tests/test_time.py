import tensorflow as tf
from timeit import default_timer

config = tf.ConfigProto()
config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0

def cpu(x):
    tf.reset_default_graph()
    with tf.device('/device:CPU:0'):
        with tf.Session(config=config) as sess:
            st = default_timer()
            tf.matmul(tf.random_normal((x,x)), tf.random_normal((x,x))).eval()
            print(default_timer() - st)

def rpc(x):
    tf.reset_default_graph()
    with tf.device('/device:RPC:0'):
        with tf.Session(config=config) as sess:
            st = default_timer()
            tf.matmul(tf.random_normal((x,x)), tf.random_normal((x,x))).eval()
            print(default_timer() - st)

cpu(200)
rpc(200)
