import tensorflow as tf
from timeit import default_timer, timeit

config = tf.ConfigProto()

def build_graph(x):
    tf.reset_default_graph()
    m = tf.matmul(tf.random_normal((x,x)), tf.random_normal((x,x)))
    return tf.reduce_sum(m)

def time_on(dev, x):
    print('Run on {}'.format(dev))
    with tf.device('/device:' + dev + ':0'):
        op = build_graph(x)
        with tf.Session(config=config) as sess:
            sess.run(op)  # warm up
            st = default_timer()
            rep = 10
            for _ in range(rep):
                sess.run(op)
            dur = (default_timer() - st) / rep
            print('Used time: {}s'.format(dur))

def gpu(x):
    return time_on('GPU', x)

def cpu(x):
    return time_on('CPU', x)

def rpc(x):
    return time_on('RPC', x)

gpu(200)
cpu(200)
rpc(200)
