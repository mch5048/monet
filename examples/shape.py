'''
this is a test about dynamic and static shapes
& rank, reduction axes.
'''

import tensorflow as tf

N, H, W, C = 10, 5, 5, 4
x = tf.get_variable(dtype=tf.float32, 
                    shape=[N, H, W, C], 
                    initializer=tf.constant_initializer(0.1), 
                    name='x')

y = tf.placeholder(dtype=tf.float32,
                   shape=[None, 3, 2],
                   name='y')

inputs_shape = x.shape
inputs_rank = x.shape.ndims

reduction_axis = inputs_rank - 1
params_shape = inputs_shape[reduction_axis: reduction_axis + 1]

print(params_shape)
print(x.dtype)

print(x.dtype.base_dtype)

a = 0.5

op = x * a

assign_op = tf.assign(x, x*a)