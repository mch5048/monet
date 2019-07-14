import numpy as np
import tensorflow as tf
from source import layers

x = np.arange(2*4*4*3).reshape([2, 4, 4, 3]).astype(np.float32)

print(x[0])
print(x[0].shape)
x = tf.convert_to_tensor(x)

out = layers.instance_normalization(inputs=x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    o = sess.run(out)
    print(o[0])
    print(o[0].shape)