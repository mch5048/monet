import tensorflow as tf
from test.src.base import prelim_ops as po
import numpy as np

x_sample = np.ones((1, 5, 5, 3))
print(x_sample)

x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 5, 5, 3], name='x_ph')

out = po.conv2d(input_=x_ph,
				filters=2,
				kernel_size=3,
				stride_size=1,
				padding='SAME',
				activation=None,
				name='conv2d_1')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	with tf.variable_scope('conv2d_1', reuse=True):
		W = sess.run(tf.get_variable('W'))
		print(W, W.shape)
		print(sess.run(tf.get_variable('b')))

	output = sess.run(out, feed_dict={x_ph: x_sample})
	print(output)
	print(output.shape)