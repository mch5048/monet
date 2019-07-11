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
                        name):
            with tf.variable_scope(name):
                # downsampling path
                out = layers.conv2d(inputs=self.datapipe.next_element,
                                    filters=64,
                                    kernel_size=3,
                                    stride_size=1,
                                    padding='VALID',
                                    activation=tf.nn.relu,
                                    name='conv1')
                out = layers.conv2d(inputs=out1,
                                    filters=64,
                                    kernel_size=3,
                                    stride_size=1,
                                    padding='VALID',
                                    activation=tf.nn.relu,
                                    name='conv2')
                maxp = layers.max_pooling2d(inputs=out1,
                                            pool_size=2,
                                            strides=2,
                                            padding='VALID')
            return out, maxp

        # FIRST: let's build this crude
        # THEN: we will factorize using .json
        def _build_graph(self):
            # downsampling path
            out1, maxp1 = self._block_down(inputs=self.datapipe.next_element,
                                           filters=64,
                                           name='block1')
            out2, maxp2 = self._block_down(inputs=self.datapipe.next_element,
                                           filters=64*2,
                                           name='block1')
            out3, maxp3 = self._block_down(inputs=self.datapipe.next_element,
                                           filters=64*4,
                                           name='block1')
            out4, maxp4 = self._block_down(inputs=self.datapipe.next_element,
                                           filters=64*8,
                                           name='block1')
            out5, maxp5 = self._block_down(inputs=self.datapipe.next_element,
                                           filters=64*16,
                                           name='block1')

            


            # upsampling path
