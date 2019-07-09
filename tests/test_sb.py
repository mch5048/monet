import tensorflow as tf
from source.misc import spatial_broadcast

tf.enable_eager_execution()

N, k = 2, 3
z = tf.get_variable('z', shape=[N, k], initializer=tf.truncated_normal_initializer())

print(z)

w, h = 5, 5
z_sb = spatial_broadcast(z=z, w=w, h=h)

print(z_sb)