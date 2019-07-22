import tensorflow as tf
from network import build_network
from source import layers
from source.misc import sampler_normal 

class VAE(object):
    def __init__(self,
                 network_specs,
                 scope='component_vae',
                 mode='training'):

        self.network_specs = network_specs
        self.scope = scope

    def __call__(self,
                 images,
                 log_mask,
                 reuse):

        with tf.variable_scope(self.scope, reuse=reuse):
            return self._build_graph(images, log_mask)
        
    def _build_graph(self, images, log_mask):
        inputs = tf.concat([images, log_mask], axis=-1)
        z_mean, z_logvar, z_samples = self._build_encoder(inputs)
        re_mask, re_image_mean = self._build_decoder(z_samples)
        return z_mean, z_logvar, re_mask, re_image_mean

    def _build_encoder(self, inputs):
        encoder = build_network(inputs=inputs,
                                model_specs=self.network_specs['encoder'],
                                latent_dim=self.network_specs['latent_dim'],
                                name='encoder')
        z_mean, z_logvar = tf.split(encoder, 
                                    [self.network_specs['latent_dim'], self.network_specs['latent_dim']], 
                                    axis=1)
        z_samples = sampler_normal(z_mean, 
                                   z_logvar)
        return z_mean, z_logvar, z_samples

    def _build_decoder(self, inputs):
        logits = build_network(inputs=inputs,
                               model_specs=self.network_specs['decoder'],
                               num_channel=self.network_specs['num_channel'],
                               name='decoder')
        re_mask, re_image_mean = tf.split(logits,
                                          [1, 3],
                                          axis=-1)

        # i think this should be trained with tf.nn.sigmoid
        # because the means should be between 0 and 1
        re_image_mean = tf.nn.sigmoid(re_image_mean)
        return re_mask, re_image_mean
        
class UNet(object):
    def __init__(self,
                 network_specs,
                 scope='unet',
                 mode='training'):

        # network specs
        self.network_specs = network_specs
        self.scope = scope
        self.mode = mode

    def __call__(self,
                 images,
                 log_scope,
                 reuse):
        with tf.variable_scope(self.scope, reuse=reuse):
            return self._build_graph(images, log_scope)

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
                  scope,
                  upsampling=True):
        with tf.variable_scope(scope):
            # upsample first
            if upsampling:
                up_inputs = layers.upsampling_2d(inputs=up_inputs,
                                           factors=[2, 2])
            out = layers.crop_to_fit(down_inputs=down_inputs,
                                     up_inputs=up_inputs)
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
    def _build_graph(self, images, log_scope):
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

        inputs = tf.concat([images, log_scope], axis=-1)

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
        shape = down_out4.get_shape().as_list()
        H, W, C = shape[1], shape[2], shape[3]
        print('down_out4: ', shape)
        print('down_out5: ', down_out5.shape)
        out = layers.flatten(down_out5)

        print('flatten shape: ', out.shape)
        out = layers.dense(inputs=out,
                           units=128,
                           activation=tf.nn.relu,
                           name='layer1')
        out = layers.dense(inputs=out,
                           units=128,
                           activation=tf.nn.relu,
                           name='layer2')
        out = layers.dense(inputs=out,
                           units=H*W*C,
                           activation=tf.nn.relu,
                           name='layer3')
        out = tf.reshape(out, shape=[-1, H, W, C])
        print('upsampling input shape: ', out.shape)

        # upsampling path
        up_out4 = self._block_up(down_inputs=down_out4,
                                 up_inputs=out,
                                 filters=init_filter*8,
                                 padding='SAME',
                                 scope='up_block4',
                                 upsampling=False)
        print('built up_out4...')
        up_out3 = self._block_up(down_inputs=down_out3,
                                 up_inputs=up_out4,
                                 filters=init_filter*4,
                                 padding='SAME',
                                 scope='up_block3')
        print('built up_out3...')
        up_out2 = self._block_up(down_inputs=down_out2,
                                 up_inputs=up_out3,
                                 filters=init_filter*2,
                                 padding='SAME',
                                 scope='up_block2')
        print('built up_out2...')
        up_out1 = self._block_up(down_inputs=down_out1,
                                 up_inputs=up_out2,
                                 filters=init_filter,
                                 padding='SAME',
                                 scope='up_block1')
        print('built up_out1...')

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

        '''
        # log softmax DOES NOT WORK PROPERLY
        # compute log_softmax for the current attention
        shape = tf.shape(logits)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        print('logits shape before: ', logits.shape)
        logits = layers.flatten(logits)
        print('logits shape: ', logits.shape)

        log_softmax = tf.nn.log_softmax(logits=logits,
                                        axis=-1)

        log_a_k = tf.reshape(log_softmax, [N, H, W, C])
        print('log_softmax shape: ', log_a_k.shape)
        '''

        log_a_k = tf.log_sigmoid(logits)
        return log_a_k