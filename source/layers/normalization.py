import tensorflow as tf

class InstanceNormalization(object):
    def __init__(self,
                 inputs,
                 epsilon=1e-6,
                 momentum=0.99,
                 name='instance_normalization'):

        self.inputs = inputs

        inputs_shape = inputs.shape
        inputs_rank = inputs.shape.ndims

        # we are only using NHWC, our reduction axis is C
        reduction_axis = inputs_rank - 1

        # will be used for moving_mean and _var, beta and gamma per C
        params_shape = inputs_shape[reduction_axis: reduction_axis + 1]

        # dtype taken from inputs: base_dtype removes _ref
        dtype = inputs.dtype.base_dtype

        # moment_axes are H and W from NHWC
        self.moment_axes = [1, 2]

        self.momentum = momentum

        self.epsilon = epsilon

        with tf.variable_scope(name):
            # those all have shape of params_shape
            self.moving_mean = tf.get_variable(dtype=dtype,
                                               shape=params_shape,
                                               initializer=tf.zeros_initializer(),
                                               name='moving_mean')
            self.moving_var = tf.get_variable(dtype=dtype,
                                              shape=params_shape,
                                              initializer=tf.ones_initializer(),
                                              name='moving_var')
            
            # gamma is scale, beta is offset
            self.gamma = tf.get_variable(dtype=dtype,
                                         shape=params_shape,
                                         initializer=tf.ones_initializer(),
                                         name='gamma') 
            self.beta = tf.get_variable(dtype=dtype,
                                        shape=params_shape,
                                        initializer=tf.zeros_initializer(),
                                        name='beta')
    def apply(self):
        # keep_dims=True will return N * 1 * 1 * C mu and var
        mean, var = tf.nn.moments(x=self.inputs, 
                                  axes=self.moment_axes,
                                  keep_dims=True)

        print(mean.get_shape().as_list(), var.get_shape().as_list())

        # here's another confusing part in tensorflow
        # correct EMA to update variable with value is alpha * value + (1 - alpha) * variable
        # we do the opposite, which is fine, technically the same things. as far as understand alpha = 1 - momentum
        # higher momentum does not forget the past easily
        # MUST CHANGE THIS
        print(self.moving_mean.dtype)

        reduced_mean = tf.reduce_mean(tf.squeeze(mean), axis=0)
        reduced_var = tf.reduce_mean(tf.squeeze(var), axis=0)
        print(reduced_mean.shape, reduced_var.shape)

        update_mean = tf.assign(self.moving_mean, self.moving_mean * self.momentum + reduced_mean * (1.0 - self.momentum))
        
        # this is also not correct moving variance calculation, BUT it's simpler and what tf uses
        update_var = tf.assign(self.moving_var, self.moving_var * self.momentum + reduced_var * (1.0 - self.momentum))

        with tf.control_dependencies([update_mean, update_var]):
            return tf.nn.batch_normalization(x=self.inputs,
                                             mean=mean,
                                             variance=var,
                                             offset=self.beta,
                                             scale=self.gamma,
                                             variance_epsilon=self.epsilon,
                                             name='instance_bn')
    def inference_apply(self):
        return tf.nn.batch_normalization(x=self.inputs,
                                         mean=self.moving_mean,
                                         variance=self.moving_var,
                                         offset=self.beta,
                                         scale=self.gamma,
                                         variance_epsilon=self.epsilon,
                                         name='instance_bn')

def instance_normalization(inputs,
                           epsilon=1e-6,
                           momentum=0.99,
                           mode='training',
                           name='instance_normalization'):
    layer = InstanceNormalization(inputs=inputs,
                                  epsilon=epsilon,
                                  momentum=momentum,
                                  name=name)
    if mode == 'training':
        return layer.apply()
    elif mode == 'evaluating':
        return layer.inference_apply()
    # we check this earlier, yet being extra careful does not hurt
    else:
        raise NotImplementedError('only training and evaluating modes are implemented')
