import tensorflow as tf
import numpy as np 

def conv2d(inputs, 
           filters, 
           kernel_size, 
           stride_size, 
           padding='SAME',
           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
           bias_initializer=tf.constant_initializer(value=0.01),
           activation=None,
           name='conv2d'):
    
    with tf.variable_scope(name):
        in_channels = inputs.get_shape()[-1]
        out_channels = filters
        W = tf.get_variable(name='W', 
                            shape=[kernel_size, kernel_size, in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=kernel_initializer)
        b = tf.get_variable(name='b', 
                            shape=[out_channels], 
                            dtype=tf.float32,
                            initializer=bias_initializer)

        conv = tf.nn.conv2d(input=inputs, 
                            filter=W, 
                            strides=[1, stride_size, stride_size, 1], 
                            padding=padding,
                            name='conv2d_')
        conv = tf.nn.bias_add(value=conv, 
                              bias=b,
                              name='bias_')
        print('conv', conv.get_shape())
        print('activation', activation)
        # only works for some activation functions
        # need to change this for general case
        if activation:
            return activation(features=conv, 
                              name='activation')
        return conv

def conv2d_transpose(inputs,
                     filters,
                     kernel_size,
                     stride_size,
                     padding='SAME',
                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                     bias_initializer=tf.constant_initializer(value=0.01),
                     activation=None,
                     name='conv2d_transpose'):
    
    if padding != 'SAME':
        raise NotImplementedError('only SAME padding is implemented for now')

    with tf.variable_scope(name):
        input_shape = inputs.get_shape()
        out_channels = filters
        
        N = tf.shape(inputs)[0]
        H, W = input_shape[1] * stride_size, input_shape[2] * stride_size
        output_shape = tf.stack([N, H, W, filters])

        W = tf.get_variable(name='W',
                            shape=[kernel_size, kernel_size, out_channels, input_shape[-1]],
                            dtype=tf.float32,
                            initializer=kernel_initializer)
        b = tf.get_variable(name='b',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=bias_initializer)
        conv = tf.nn.conv2d_transpose(value=inputs,
                                      filter=W,
                                      output_shape=output_shape,
                                      strides=[1, stride_size, stride_size, 1],
                                      padding=padding,
                                      name='conv2d_transpose_')
        conv = tf.nn.bias_add(value=conv,
                              bias=b,
                              name='bias_')

        # only works for some activation functions
        # need to change this for general case
        if activation:
            return activation(features=conv,
                              name='activation')
        return conv

def max_pooling2d(inputs,
                  pool_size,
                  strides,
                  padding='VALID',
                  name='max_pooling2d'):
    with tf.name_scope(name):
        max_pool = tf.nn.max_pool(value=inputs,
                                  ksize=[1, pool_size, pool_size, 1],
                                  strides=[1, strides, strides, 1],
                                  padding=padding,
                                  name='max_pooling2d_')

    print('max_pool', max_pool.get_shape())
    return max_pool

def dropout(inputs, 
            rate=0.5, 
            noise_shape=None, 
            seed=None,
            name='dropout'):
    return tf.nn.dropout(x=inputs,
                         keep_prob=1.0 - rate,
                         noise_shape=noise_shape,
                         seed=seed,
                         name=name)

def dense(inputs, 
          units, 
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
          bias_initializer=tf.constant_initializer(value=0.01), 
          activation=None,
          name='fc'):
    with tf.variable_scope(name):
        in_shape = inputs.get_shape()[1]
        W = tf.get_variable(name='W', 
                            shape=[in_shape, units], 
                            dtype=tf.float32,
                            initializer=kernel_initializer)
        b = tf.get_variable(name='b',
                            shape=[units],
                            dtype=tf.float32,
                            initializer=bias_initializer)
        dense = tf.add(tf.matmul(inputs, W), b)
        
        if activation:
            return activation(features=dense, 
                              name='activation')
        return dense

def flatten(inputs, name='flatten'):
    with tf.name_scope(name):
        # [batch_size, ...]
        shape = inputs.get_shape()
        out = np.product(shape[1:])
        return tf.reshape(tensor=inputs, 
                          shape=[-1, out], 
                          name=name)

# spatial broadcast operator
def spatial_broadcast(z, w, h, name='spatial_broadcast'):
    with tf.name_scope(name):
        # look at dynamic vs static shape
        batch_size = tf.shape(z)[0]
        k = z.get_shape()[-1]

        z_b = tf.tile(input=z, multiples=[1, h * w], name='tile')
        z_b = tf.reshape(z_b, shape=[batch_size, h, w, k])

        print('in spatial broadcast', z_b.get_shape())

        '''
        indexing does NOT matter
        1. square output (because we have square images) 2. channels are order invariant
        '''
        x = tf.linspace(-1.0, 1.0, num=w)
        y = tf.linspace(-1.0, 1.0, num=w)
        x_b, y_b = tf.meshgrid(x, y)
        
        # we need (w, w, 1)
        x_b = tf.expand_dims(x_b, axis=-1)
        y_b = tf.expand_dims(y_b, axis=-1)

        # apply concat to each sample in z_b
        z_sb = tf.map_fn(fn=lambda z_i: tf.concat([z_i, x_b, y_b], axis=-1), 
                         elems=z_b,
                         parallel_iterations=True,
                         back_prop=True,
                         swap_memory=True,
                         name='map_concat')
        print(z_sb.get_shape())
    return z_sb

# we will be using transposed convolution weights with 'same' padding
# cheap and faster solution for nearest neighbors upsampling
def upsampling_2d(inputs,
                  factor,
                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                  name='upsampling_2d'):
    # i should just use tf.image.resize_images with NEAREST_NEIGHBORS
    with tf.name_scope(name):
        input_shape = inputs.get_shape()
        H, W = input_shape[1] * factor, input_shape[2] * factor
        images = tf.image.resize_images(images=inputs,
                                        size=[H, W],
                                        method=method,
                                        align_corners=True,
                                        preserve_aspect_ratio=False)
    return images

def crop_to_fit(down_input, 
                up_input,
                name='crop_to_fit'):
    with tf.name_scope(name):
        # get shapes
        down_shape = down_input.get_shape()
        up_shape = up_input.get_shape()

        print(down_shape, up_shape)

        down_H, down_W = down_shape[1], down_shape[2]
        up_H, up_W = up_shape[1], up_shape[2]

        # extract the center (the reason for subtracting 1)
        d_H = down_H - up_H - 1
        print('d_H in this level is: {}'.format(d_H + 1))

        d_W = down_W - up_W - 1
        print('d_W in this level is: {}'.format(d_W + 1))

        down_input = down_input[:, d_H:(d_H + up_H), d_W:(d_W + up_W), :]
        print('down_input cropped shape: ', down_input.get_shape())
        print(np.squeeze(down_input))
        print('up_input shape: ', up_input.get_shape())

        concat = tf.concat([down_input, up_input], axis=-1)
        print('concat shape: ', concat.get_shape())

    return concat