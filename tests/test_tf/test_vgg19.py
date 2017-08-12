from __future__ import print_function

import unittest

import tensorflow as tf

from . import run_on_rpc_and_cpu, run_on_devices
from . import networks, datasets


def run_vgg19(sess, input_data):
    images, labels, num_classes = input_data(batch_size=100, batch_num=None)
    train_mode = tf.placeholder(tf.bool)

    vgg = networks.Vgg19Trainable()
    vgg.build(images, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=vgg.prob))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(vgg.prob, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    for i in range(50):
        sess.run(train_step, feed_dict={train_mode: True})

    return sess.run(accuracy, feed_dict={train_mode: False})


class TestVGG19(unittest.TestCase):
    def test_cpu(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg19(sess, input_data)

        run_on_devices(func, '/device:CPU:0')

    def test_fake_data(self):
        def func():
            def input_data(*a, **kw):
                return datasets.fake_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg19(sess, input_data)

        actual, expected = run_on_rpc_and_cpu(func)
        self.assertEquals(actual, expected)

    def test_flowers(self):
        def func():
            def input_data(*a, **kw):
                return datasets.flowers_data(*a, height=224, width=224, num_classes=1000, **kw)
            sess = tf.get_default_session()
            return run_vgg19(sess, input_data)

        actual, expected = run_on_rpc_and_cpu(func)
        self.assertEquals(actual, expected)


if __name__ == '__main__':
    unittest.main()
