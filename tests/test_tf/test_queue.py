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
from __future__ import print_function, division

import unittest
import tensorflow as tf
import numpy as np

from parameterized import parameterized

from . import run_on_devices, run_on_rpc_and_cpu, assertAllClose
from .lib import tfhelper

def run_queue(sess):
    # Create a slightly shuffled batch of integers in [0, 100)
    queue = tf.train.range_input_producer(limit=100, num_epochs=1, shuffle=True)

    batch_num = 10
    get_batch = queue.dequeue_many(100 // batch_num)

    preds = []
    with tfhelper.initialized_scope(sess) as coord:
        for i in range(batch_num):
            if coord.should_stop():
                break
            val = sess.run(get_batch)
            print("batch {}, got {}".format(i, val))
            preds.append(val)

    return preds


class TestQueue(unittest.TestCase):
    def setUp(self):
        # Called prior to each test method
        pass

    def tearDown(self):
        # Called after each test method
        pass

    def test_cpu(self):
        def func():
            sess = tf.get_default_session()
            return run_queue(sess)

        run_on_devices(func, '/device:CPU:0')

    def test_random_shuffle(self):
        def func():
            sess = tf.get_default_session()
            return run_queue(sess)

        actual, expected = run_on_rpc_and_cpu(func)
        assertAllClose(actual, expected)


if __name__ == "__main__":
    unittest.main()
