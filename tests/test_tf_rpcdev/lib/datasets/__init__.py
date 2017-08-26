from __future__ import absolute_import, division, print_function

import os
import multiprocessing
import tensorflow as tf

from . import flowers
from .. import tfhelper

slim = tf.contrib.slim


def fake_data(batch_size, batch_num, is_train=True, height=256, width=256, num_classes=1000):
    """Generate a fake dataset that matches the dimensions of ImageNet."""
    if not is_train:
        batch_num = 1
        batch_size = 1
    with tf.name_scope('fake_data'):
        image = tf.Variable(tf.random_normal([height, width, 3], dtype=tf.float32),
                            name='sample_image', trainable=False)
        label = tf.Variable(tf.random_uniform([1], minval=0, maxval=1000,
                                              dtype=tf.int32),
                            name='ground_truth', trainable=False)

        image_queue = tf.train.input_producer(tf.expand_dims(image, 0),
                                              num_epochs=batch_num * batch_size)
        label_queue = tf.train.input_producer(tf.expand_dims(label, 0),
                                              num_epochs=batch_num * batch_size)

        images = image_queue.dequeue_many(batch_size, name='images')
        labels = label_queue.dequeue_many(batch_size, name='labels')
        labels = tf.squeeze(labels)  # remove the second dim (10, 1) => (10, )
        return images, labels, num_classes


def flowers_data(batch_size, batch_num, is_train=True, num_threads=None):
    """Flowers dataset from Facebook"""
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
        num_readers = num_threads // 2

    with tf.name_scope('flowers_data'):
        data_dir = os.path.join(os.path.expanduser('~'), 'data', 'flowers')
        if is_train:
            dataset = flowers.get_split('train', data_dir)
        else:
            dataset = flowers.get_split('validation', data_dir)

        if is_train:
            num_epochs = (batch_num * batch_size + dataset.num_samples - 1) // dataset.num_samples
        else:
            num_epochs = 1
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=num_readers,
                                                                  num_epochs=num_epochs)
        image, label = provider.get(['image', 'label'])
        tf.image_summary('image', tf.expand_dims(image, 0))
        # Transform the image to floats.
        image = tf.to_float(image)

        # Resize and crop if needed.
        resized_image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
        tf.image_summary('resized_image', tf.expand_dims(resized_image, 0))

        # Subtract off the mean and divide by the variance of the pixels.
        resized_image = tfhelper.image_standardization(resized_image)

        images, labels = tf.train.batch([resized_image, label], batch_size=batch_size,
                                        capacity=1000 * batch_size, num_threads=num_readers)
        return images, labels, 5
