import tensorflow as tf
from source import layers

class UNet(object):
    def __init__(self,
                 network_spec,
                 scope='unet',
                 mode='training',
                 datapipe=None,
                 training_params=None):

        # network specs
        self.network_specs = network_specs

        if mode == 'training':

            # datapipe
            self.datapipe = datapipe
            self.inputs = datapipe.images
            self.input_shape = list(self.inputs.shape[1:])
            
            # training_params has all the training parameters
            self.lr = training_params['lr']
            self.loss = training_params['loss']
            self.n_run = datapipe.n_run

            with tf.variable_scope(scope):
                self._build_graph()
                self._build_loss()
                self._build_optimizer()
            self.vars_initializer = tf.global_variables_initializer()

        if mode == 'evaluating':
            # TODO
            pass

        # saver
        self.saver = tf.train.Saver()
 
        # config for session    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49,
                                    allow_growth=True)

        self.config = tf.ConfigProto(gpu_options=gpu_options,
                                     inter_op_parallelism_threads=4,
                                     intra_op_parallelism_threads=4,
                                     allow_soft_placement=True)

        def _block_down(self,
                        inputs, 
                        filters,
                        padding,
                        name):
            with tf.variable_scope(name):
                # downsampling path
                out = layers.conv2d(inputs=inputs,
                                    filters=filters,
                                    kernel_size=3,
                                    stride_size=1,
                                    padding=padding,
                                    activation=tf.nn.relu,
                                    name='conv1')
                out = layers.conv2d(inputs=out,
                                    filters=filters,
                                    kernel_size=3,
                                    stride_size=1,
                                    padding=padding,
                                    activation=tf.nn.relu,
                                    name='conv2')
                maxp = layers.max_pooling2d(inputs=out1,
                                            pool_size=2,
                                            strides=2,
                                            padding='VALID')
            return out, maxp

        def _block_up(down_inputs,
                      up_inputs,
                      filters,
                      padding,
                      name):
            with tf.variable_scope(name):
                # upsample first
                out = layers.upsampling_2d(inputs=up_inputs,
                                           factors=[2, 2])
                out = layers.crop_to_fit(down_input=down_inputs,
                                         up_inputs=out)
                out = layers.conv2d(inputs=out,
                                    filters=filters,
                                    kernel_size=3,
                                    stride_size=1,
                                    padding=padding,
                                    activation=tf.nn.relu,
                                    name='conv1')
                out = layers.conv2d(inputs=out,
                                    filters=filters,
                                    kernel_size=3,
                                    stride_size=1,
                                    padding=padding,
                                    activation=tf.nn.relu,
                                    name='conv2')
            return out

        # FIRST: let's build this crude
        # changed padding from 'VALID' to 'SAME' to obtain the same output shape
        # we will be testing unet segmentation on dspirites
        # THEN: we will factorize using .json
        def _build_graph(self):
            # downsampling path
            down_out1, maxp1 = self._block_down(inputs=self.datapipe.next_element,
                                                filters=64,
                                                name='down_block1')
            down_out2, maxp2 = self._block_down(inputs=maxp1,
                                                filters=64*2,
                                                name='down_block2')
            down_out3, maxp3 = self._block_down(inputs=maxp2,
                                                filters=64*4,
                                                name='down_block3')
            down_out4, maxp4 = self._block_down(inputs=maxp3,
                                                filters=64*8,
                                                name='down_block4')
            down_out5, maxp5 = self._block_down(inputs=maxp4,
                                                filters=64*16,
                                                name='down_block5')
            up_out1





            # upsampling path
