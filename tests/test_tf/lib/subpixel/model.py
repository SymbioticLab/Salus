from __future__ import division, absolute_import, print_function

import tensorflow as tf

from .subpixel import PS
from . import ops


class DCGAN(object):
    def __init__(self, inputs, target_images,
                 batch_size=64, gf_dim=64, learning_rate=0.0002, beta1=0.5):
        """

        Args:
            batch_size: The size of batch. Should be specified before training.
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        """
        self.batch_size = batch_size

        # input variables
        self.inputs = inputs
        self.target_images = target_images

        self.gf_dim = gf_dim

        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.G = None
        self.g_loss = None
        self.g_vars = []
        self.g_optim = None
        self.h0 = None
        self.h0_w = None
        self.h0_b = None
        self.h1 = None
        self.h1_w = None
        self.h1_b = None
        self.h2_w = None
        self.h2_b = None

    def build_model(self):
        self.G = self.generator(self.inputs)

        self.g_loss = tf.reduce_mean(tf.square(self.target_images-self.G))

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

    def generator(self, z):
        # project `z` and reshape
        self.h0, self.h0_w, self.h0_b = ops.deconv2d(z, [self.batch_size, 32, 32, self.gf_dim],
                                                     k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0', with_w=True)
        h0 = ops.lrelu(self.h0)

        self.h1, self.h1_w, self.h1_b = ops.deconv2d(h0, [self.batch_size, 32, 32, self.gf_dim],
                                                     name='g_h1', d_h=1, d_w=1, with_w=True)
        h1 = ops.lrelu(self.h1)

        h2, self.h2_w, self.h2_b = ops.deconv2d(h1, [self.batch_size, 32, 32, 3*16],
                                                d_h=1, d_w=1, name='g_h2', with_w=True)
        h2 = PS(h2, 4, color=True)

        return tf.nn.tanh(h2)
