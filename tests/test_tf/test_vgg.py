from __future__ import print_function

import unittest
from datetime import datetime
from timeit import default_timer

import tensorflow as tf

from . import run_on_rpc_and_cpu, run_on_rpc_and_gpu, run_on_sessions, run_on_devices
from . import networks, datasets
from .lib import tfhelper


def run_vgg(vgg, sess, input_data):
    if vgg is None:
        raise ValueError('VGG is None!!')

    batch_size = 100
    images, labels, num_classes = input_data(batch_size=batch_size, batch_num=100)
    train_mode = tf.placeholder(tf.bool)

    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print('Total number of variables: {}'.format(vgg.get_var_count()))

    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.prob))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # vgg.prob: [batch_size, 1000]
    # labels: [batch_size,]
    correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        print(qr.name)

    sess.run(tfhelper.initialize_op())
    coord = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(sess, coord)
    print('{} threads started for queue'.format(len(queue_threads)))
    speeds = []
    for i in range(5):
        if coord.should_stop():
            break
        print("Start running step {}".format(i))
        start_time = default_timer()
        _, loss_value = sess.run([train_step, cross_entropy], feed_dict={train_mode: True})
        duration = default_timer() - start_time

        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        speeds.append(examples_per_sec)
        fmt_str = '{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch'
        print(fmt_str.format(datetime.now(), i, loss_value, examples_per_sec, sec_per_batch))

    final_acc = sess.run(accuracy, feed_dict={train_mode: False})
    coord.request_stop()
    coord.join(queue_threads)

    return final_acc


class VGGCaseBase(unittest.TestCase):
    def _vgg(self):
        return None

    def test_gpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)

        run_on_devices(func, '/device:GPU:0', config=tf.ConfigProto(allow_soft_placement=True))

    def test_cpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)

        run_on_devices(func, '/device:CPU:0')

    def test_rpc_only(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)

        config=tf.ConfigProto()
        config.zmq_options.resource_map.temporary['MEMORY:GPU'] = 11494955340
        config.zmq_options.resource_map.persistant['MEMORY:GPU'] = 1.67e9
        run_on_sessions(func, 'zrpc://tcp://127.0.0.1:5501', config=config)

    def test_fake_data(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)

        actual, expected = run_on_rpc_and_cpu(func, config=tf.ConfigProto(allow_soft_placement=True))
        self.assertEquals(actual, expected)

    @unittest.skip("Not yet implemented")
    def test_fake_data_gpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)

        actual, expected = run_on_rpc_and_gpu(func)
        self.assertEquals(actual, expected)

    def test_flowers(self):
        def func():
            def input_data(*a, **kw):
                return datasets.flowers_data(*a, height=224, width=224, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)

        actual, expected = run_on_rpc_and_cpu(func)
        self.assertEquals(actual, expected)

    @unittest.skip("Skip distributed runtime")
    def test_dist(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)

        run_on_sessions(func, 'grpc://localhost:2222')

    @unittest.skip("Skip distributed runtime")
    def test_dist_gpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg(self._vgg(), sess, input_data)
        run_on_devices(func, '/job:local/task:0/device:GPU:0', target='grpc://localhost:2222',
                       config=tf.ConfigProto(allow_soft_placement=True))


class TestVgg11(VGGCaseBase):
    def _vgg(self):
        return networks.Vgg11Trainable()


class TestVgg16(VGGCaseBase):
    def _vgg(self):
        return networks.Vgg16Trainable()


class TestVgg19(VGGCaseBase):
    def _vgg(self):
        return networks.Vgg19Trainable()


del VGGCaseBase


if __name__ == '__main__':
    unittest.main()
