import tensorflow as tf
from network import build_network
from source import layers
from source.misc import sampler_normal 

class VAE(object):
    def __init__(self,
                 network_specs,
                 images,
                 log_mask,
                 scope='component_vae',
                 mode='training',
                 reuse=False):

        self.network_specs = network_specs
        self.images = images
        self.log_mask = log_mask
        
        with tf.variable_scope(scope, reuse=reuse):
            self._build_graph()

    def _build_graph(self):
        inputs = tf.concat([self.images, self.log_mask], axis=-1)
        z_samples = self._build_encoder(inputs)
        self._build_decoder(z_samples)

    def _build_encoder(self, inputs):
        encoder = build_network(inputs=inputs,
                                model_specs=self.network_specs['encoder'],
                                latent_dim=self.network_specs['latent_dim'],
                                name='encoder')
        self.mean, self.log_var = tf.split(encoder, 
                                           [self.network_specs['latent_dim'], self.network_specs['latent_dim']], 
                                           axis=1)
        z_samples = sampler_normal(self.mean, 
                                   self.log_var)
        return z_samples

    def _build_decoder(self, inputs):
        logits = build_network(inputs=inputs,
                               model_specs=self.network_specs['decoder'],
                               num_channel=self.network_specs['num_channel'],
                               name='decoder')
        self.log_mask, image_mean = tf.split(logits,
                                             [1, 3],
                                             axis=-1)
        
        self.image_mean = tf.nn.sigmoid(image_mean)


    def output(self):
        return self.mean, self.log_var, self.log_mask, self.image_mean

class UNet(object):
    def __init__(self,
                 network_specs,
                 images,
                 log_scope,
                 scope='unet',
                 mode='training',
                 reuse=False):

        # network specs
        self.network_specs = network_specs
        self.mode = mode

        self.images = images
        self.log_scope = log_scope

        with tf.variable_scope(scope, reuse=reuse):
            self._build_graph()

    # THIS FUNCTIONS WILL BE REMOVED
    # _block_down and _block_up
    def _block_down(self,
                    inputs, 
                    filters,
                    padding,
                    scope):
        with tf.variable_scope(scope):
            # downsampling path
            out = layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=3,
                                stride_size=1,
                                padding=padding,
                                normalization='instance_normalization',
                                activation=tf.nn.relu,
                                mode=self.mode,
                                name='conv1')
            out = layers.conv2d(inputs=out,
                                filters=filters,
                                kernel_size=3,
                                stride_size=1,
                                padding=padding,
                                normalization='instance_normalization',
                                activation=tf.nn.relu,
                                mode=self.mode,
                                name='conv2')
            maxp = layers.max_pooling2d(inputs=out,
                                        pool_size=2,
                                        strides=2,
                                        padding='VALID')
        return out, maxp

    def _block_up(self,
                  down_inputs,
                  up_inputs,
                  filters,
                  padding,
                  scope):
        with tf.variable_scope(scope):
            # upsample first
            out = layers.upsampling_2d(inputs=up_inputs,
                                       factors=[2, 2])
            out = layers.crop_to_fit(down_inputs=down_inputs,
                                     up_inputs=out)
            out = layers.conv2d(inputs=out,
                                filters=filters,
                                kernel_size=3,
                                stride_size=1,
                                padding=padding,
                                normalization='instance_normalization',
                                activation=tf.nn.relu,
                                mode=self.mode,                                
                                name='conv1')
            out = layers.conv2d(inputs=out,
                                filters=filters,
                                kernel_size=3,
                                stride_size=1,
                                padding=padding,
                                normalization='instance_normalization',
                                activation=tf.nn.relu,
                                mode=self.mode,
                                name='conv2')
        return out

    # FIRST: let's build this crude
    # changed padding from 'VALID' to 'SAME' to obtain the same output shape
    # we will be testing unet segmentation on dspirites
    # THEN: we will factorize using .json
    def _build_graph(self):
        '''
        # initial log scope
        if self.log_scope is None:
            shape = self.images.get_shape().as_list()
            B, H, W, C = 16, shape[1], shape[2], 1
            # np.ones([self.datapipe.batch_size, H, W, C]).astype(np.float32)
            self.log_scope = tf.get_variable('log_scope0', 
                                             shape=[B, H, W, C], 
                                             initializer=tf.zeros_initializer(), 
                                             dtype=tf.float32, 
                                             trainable=False)
        '''

        inputs = tf.concat([self.images, self.log_scope], axis=-1)

        # downsampling path
        init_filter = 16
        down_out1, maxp1 = self._block_down(inputs=inputs,
                                            filters=init_filter,
                                            padding='SAME',
                                            scope='down_block1')
        down_out2, maxp2 = self._block_down(inputs=maxp1,
                                            filters=init_filter*2,
                                            padding='SAME',
                                            scope='down_block2')
        down_out3, maxp3 = self._block_down(inputs=maxp2,
                                            filters=init_filter*4,
                                            padding='SAME',
                                            scope='down_block3')
        down_out4, maxp4 = self._block_down(inputs=maxp3,
                                            filters=init_filter*8,
                                            padding='SAME',
                                            scope='down_block4')
        down_out5, maxp5 = self._block_down(inputs=maxp4,
                                            filters=init_filter*16,
                                            padding='SAME',
                                            scope='down_block5')

        # they put a 3-layer MLP here
        
        # upsampling path
        up_out4 = self._block_up(down_inputs=down_out4,
                                 up_inputs=down_out5,
                                 filters=init_filter*8,
                                 padding='SAME',
                                 scope='up_block4')
        up_out3 = self._block_up(down_inputs=down_out3,
                                 up_inputs=up_out4,
                                 filters=init_filter*4,
                                 padding='SAME',
                                 scope='up_block3')
        up_out2 = self._block_up(down_inputs=down_out2,
                                 up_inputs=up_out3,
                                 filters=init_filter*2,
                                 padding='SAME',
                                 scope='up_block2')
        up_out1 = self._block_up(down_inputs=down_out1,
                                 up_inputs=up_out2,
                                 filters=init_filter,
                                 padding='SAME',
                                 scope='up_block1')

        # final layers
        ## TODO
        logits = layers.conv2d(inputs=up_out1,
                               filters=1,
                               kernel_size=1,
                               stride_size=1,
                               padding='SAME',
                               normalization=None,
                               activation=None,
                               mode=self.mode,
                               name='final_layer')

        # compute log_softmax for the current attention
        # CAVEAT = axis[1, 2], changed it
        shape = tf.shape(logits)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        print('logits shape before: ', shape)
        logits = tf.reshape(tf.transpose(logits, [0, 3, 1, 2]), [N * C, H * W])
        print('logits shape: ', logits.shape)

        log_softmax = tf.nn.log_softmax(logits=logits,
                                        axis=-1)
        log_neg = tf.log(1.0 - tf.exp(log_softmax))

        self.log_softmax = tf.transpose(tf.reshape(log_softmax, [N, C, H, W]), [0, 2, 3, 1])
        self.log_neg = tf.transpose(tf.reshape(log_neg, [N, C, H, W]), [0, 2, 3, 1])

        print('log_softmax shape: ', self.log_softmax.shape)
        print('log_neg shape: ', self.log_neg.shape)

    def output(self):
        log_mask = self.log_scope + self.log_softmax
        log_scope = self.log_scope + self.log_neg
        return log_mask, log_scope

    def additional_output(self):
        return self.log_softmax, self.log_neg