from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def initialize_op():
    """Operation to initialize global and local variables"""
    if hasattr(tf, 'global_variables_initializer'):
        # tensorflow 0.12
        return tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
    else:
        return tf.group(tf.initialize_all_variables(),
                        tf.initialize_local_variables())


def global_variables():
    if hasattr(tf, 'global_variables'):
        return tf.global_variables()
    else:
        return tf.all_variables()


def scalar_summary(*args, **kwargs):
    if hasattr(tf.summary, 'scalar'):
        return tf.summary.scalar(*args, **kwargs)
    else:
        return tf.scalar_summary(*args, **kwargs)


def histogram_summary(*args, **kwargs):
    if hasattr(tf.summary, 'histogram'):
        return tf.summary.histogram(*args, **kwargs)
    else:
        return tf.histogram_summary(*args, **kwargs)


def merge_all_summaries(*args, **kwargs):
    if hasattr(tf.summary, 'merge_all'):
        return tf.summary.merge_all(*args, **kwargs)
    else:
        return tf.merge_all_summaries(*args, **kwargs)


def image_standardization(image):
    if hasattr(tf.image, 'per_image_standardization'):
        return tf.image.per_image_standardization(image)
    else:
        return tf.image.per_image_whitening(image)
