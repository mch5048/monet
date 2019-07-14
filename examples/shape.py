'''
this is a test about dynamic and static shapes
'''

import tensorflow as tf

x = tf.get_variable(dtype=tf.float32, 
                    shape=[3, 2], 
                    initializer=tf.constant_initializer(0.1), 
                    name='x')

y = tf.placeholder(dtype=tf.float32,
                   shape=[None, 3, 2],
                   name='y')


print(x.get_shape())
print(y.get_shape())

sh_x = tf.shape(x)
sh_y = tf.shape(y)

print(x.shape)
print(y.shape)

# i understood