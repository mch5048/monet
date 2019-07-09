'''
simple test file to understand tf.data.Dataset and iterators
more complicated examples will be added
'''

import numpy as np
import tensorflow as tf

# what i am wondering is about how those shuffle and batch work

class Load(object):
    def __init__(self, inputs):
        self.inputs = inputs
        self._build_dataset()

    def _build_dataset(self):
        self.x_ph = tf.placeholder(self.inputs.dtype, self.inputs.shape)
        dataset = tf.data.Dataset.from_tensor_slices(self.x_ph)
        dataset = dataset.shuffle(buffer_size=15)
        # dataset = dataset.repeat(count=2)
        def s(item):
            return item + 30
        dataset = dataset.map(s, num_parallel_calls=8)
        dataset = dataset.batch(batch_size=5)
        dataset = dataset.prefetch(buffer_size=5)
        iterator = dataset.make_initializable_iterator()
        self.initializer = iterator.initializer
        self.next_element = iterator.get_next()
        item = 'fun'
        self.i = 'notfun'

# this is my data
# x = np.arange(20).reshape([2, 5, 2, 1])
x = np.arange(20)
data = Load(x)
try:
    print(data.item)
except:
    print(data.i)
with tf.Session() as sess:
    sess.run(data.initializer, feed_dict={data.x_ph: x})

    for e in range(5):
        try:
            while True:
                l = sess.run(data.next_element)
                print(l)
        except tf.errors.OutOfRangeError:
            print('we are out of range...')
            sess.run(data.initializer, feed_dict={data.x_ph: x})