#
# <one line to give the library's name and an idea of what it does.>
# Copyright (C) 2017  Aetf <aetf@unlimitedcodeworks.xyz>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function

import unittest
import tensorflow as tf
import numpy as np

from parameterized import parameterized

from . import run_on_rpc_and_cpu, assertAllClose
from .lib import tfhelper

def run_queue(sess):
    def simple_shuffle_batch(source, capacity, batch_size=10):
        # Create a random shuffle queue.
        queue = tf.RandomShuffleQueue(capacity=capacity,
                                      min_after_dequeue=int(0.9*capacity),
                                      shapes=source.shape, dtypes=source.dtype)

        # Create an op to enqueue one item.
        enqueue = queue.enqueue(source)

        # Create a queue runner that, when started, will launch threads applying
        # that enqueue op.
        num_threads = 1
        qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)

        # Register the queue runner so it can be found and started by
        # `tf.train.start_queue_runners` later (the threads are not launched yet).
        tf.train.add_queue_runner(qr)

        # Create an op to dequeue a batch
        return queue.dequeue_many(batch_size)

    # create a dataset that counts from 0 to 99
    input = tf.constant(list(range(100)))
    input = tf.contrib.data.Dataset.from_tensor_slices(input)
    input = input.make_one_shot_iterator().get_next()

    # Create a slightly shuffled batch from the sorted elements
    get_batch = simple_shuffle_batch(input, capacity=20)

    sess.run(tfhelper.initialize_op())
    coord = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(sess, coord)

    preds = []
    while not coord.should_stop():
        val = sess.run(get_batch)
        preds.append(val)

    coord.request_stop()
    coord.join(queue_threads)

    return preds


class TestQueue(unittest.TestCase):
    def setUp(self):
        # Called prior to each test method
        pass

    def tearDown(self):
        # Called after each test method
        pass

    def test_random_shuffle(self):
        def func():
            sess = tf.get_default_session()
            return run_queue(sess)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)


if __name__ == "__main__":
    unittest.main()
