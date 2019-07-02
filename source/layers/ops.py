import tensorflow as tf
import numpy as np 

def conv2d(input_, 
           filters, 
           kernel_size, 
           stride_size, 
           padding='VALID',
           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
           bias_initializer=tf.constant_initializer(value=0.01),
           activation=None,
           name='conv2d'):
    with tf.variable_scope(name):
        in_channels = input_.get_shape()[-1]
        out_channels = filters
        W = tf.get_variable(name='W', 
                            shape=[kernel_size, kernel_size, in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=kernel_initializer)
        b = tf.get_variable(name='b', 
                            shape=[out_channels], 
                            dtype=tf.float32,
                            initializer=bias_initializer)

        conv = tf.nn.conv2d(input=input_, 
                            filter=W, 
                            strides=[1, stride_size, stride_size, 1], 
                            padding=padding,
                            name='conv2d_')
        conv = tf.nn.bias_add(value=conv, 
                              bias=b,
                              name='bias_')

        if activation:
            return activation(features=conv, 
                              name='activation')
        return conv

def max_pooling2d(input_,
                  pool_size,
                  strides,
                  padding='VALID',
                  name='max_pooling2d'):
    return tf.nn.max_pool(value=input_,
                          ksize=[1, pool_size, pool_size, 1],
                          strides=[1, strides, strides, 1],
                          padding=padding,
                          name='max_pooling2d_')

def dropout(input_, 
            rate=0.5, 
            noise_shape=None, 
            seed=None,
            name='dropout'):
    return tf.nn.dropout(x=input_,
                         keep_prob=1.0 - rate,
                         noise_shape=noise_shape,
                         seed=seed,
                         name=name)

def fc(input_, 
       units, 
       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
       bias_initializer=tf.constant_initializer(value=0.01), 
       activation=None,
       name='fc'):
    with tf.variable_scope(name):
        in_shape = input_.get_shape()[1]
        W = tf.get_variable(name='W', 
                            shape=[in_shape, units], 
                            dtype=tf.float32,
                            initializer=kernel_initializer)
        b = tf.get_variable(name='b',
                            shape=[units],
                            dtype=tf.float32,
                            initializer=bias_initializer)
        dense = tf.add(tf.matmul(input_, W), b)
        
        if activation:
            return activation(features=dense, 
                              name='activation')
        return dense

def flatten(input_, name='flatten'):
    # [batch_size, ...]
    shape = input_.get_shape()
    out = np.product(shape[1:])
    return tf.reshape(tensor=input_, 
                      shape=[-1, out], 
                      name=name)
