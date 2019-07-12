import os

import numpy as np
import tensorflow as tf

from source import layers
from source import misc

class UNet(object):
    def __init__(self,
                 network_specs,
                 datapipe,
                 scope='unet',
                 mode='training',
                 training_params=None):

        # network specs
        self.network_specs = network_specs

        # hard-coded for now, change it LATER
        self.num_classes = 3

        # datapipe
        self.datapipe = datapipe

        # inputs and images are kind of confusing to use at the same time
        # might need to change to a unified name
        self.inputs = datapipe.images
        self.input_shape = list(self.inputs.shape[1:])
        self.labels = datapipe.labels

        # training mode must supply training_params
        if mode == 'training':
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
            with tf.variable_scope(scope):
                self._build_graph()

        # saver
        self.saver = tf.train.Saver()
 
        # config for session    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49,
                                    allow_growth=True)

        self.config = tf.ConfigProto(gpu_options=gpu_options,
                                     inter_op_parallelism_threads=4,
                                     intra_op_parallelism_threads=4,
                                     allow_soft_placement=True)

    # THIS FUNCTIONS WILL BE REMOVED
    # _block_down and _block_up
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
                  name):
        with tf.variable_scope(name):
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
        init_filter = 16
        self.next_images = self.datapipe.next_images
        down_out1, maxp1 = self._block_down(inputs=self.next_images,
                                            filters=init_filter,
                                            padding='SAME',
                                            name='down_block1')
        down_out2, maxp2 = self._block_down(inputs=maxp1,
                                            filters=init_filter*2,
                                            padding='SAME',
                                            name='down_block2')
        down_out3, maxp3 = self._block_down(inputs=maxp2,
                                            filters=init_filter*4,
                                            padding='SAME',
                                            name='down_block3')
        down_out4, maxp4 = self._block_down(inputs=maxp3,
                                            filters=init_filter*8,
                                            padding='SAME',
                                            name='down_block4')
        down_out5, maxp5 = self._block_down(inputs=maxp4,
                                            filters=init_filter*16,
                                            padding='SAME',
                                            name='down_block5')
        # upsampling path
        up_out4 = self._block_up(down_inputs=down_out4,
                                 up_inputs=down_out5,
                                 filters=init_filter*8,
                                 padding='SAME',
                                 name='up_block4')
        up_out3 = self._block_up(down_inputs=down_out3,
                                 up_inputs=up_out4,
                                 filters=init_filter*4,
                                 padding='SAME',
                                 name='up_block3')
        up_out2 = self._block_up(down_inputs=down_out2,
                                 up_inputs=up_out3,
                                 filters=init_filter*2,
                                 padding='SAME',
                                 name='up_block2')
        up_out1 = self._block_up(down_inputs=down_out1,
                                 up_inputs=up_out2,
                                 filters=init_filter,
                                 padding='SAME',
                                 name='up_block1')

        # final layers
        ## TODO
        self.logits = layers.conv2d(inputs=up_out1,
                                    filters=self.num_classes,
                                    kernel_size=1,
                                    stride_size=1,
                                    padding='SAME',
                                    activation=None,
                                    name='final_layer')
        self.preds = tf.nn.sigmoid(self.logits)
        self.next_labels = self.datapipe.next_labels

    def _build_loss(self):
        # need to change labels to actual labels
        if self.loss == 'cross_entropy':
            rec_loss = misc.cross_entropy(logits=self.logits, labels=self.next_labels)
        elif self.loss == 'mse':
            rec_loss = misc.mse(preds=preds, labels=self.next_labels)
        else:
            raise NotImplementError('loss: {} is not implemented.'.format(self.loss))
        self.loss = rec_loss

    def _build_optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, save_path, ckpt_path=None):
        with tf.Session(config=self.config) as sess:
            # init ops
            if ckpt_path:
                print('restoring {}...'.format(ckpt_path))
                self.saver.restore(sess, ckpt_path)
                print('restored')
            else:
                sess.run(self.vars_initializer)

            # datapipe initializer
            sess.run(self.datapipe.initializer, 
                     feed_dict={self.datapipe.images_ph: self.inputs,
                                self.datapipe.labels_ph: self.labels})

            # n_epoch, epoch loss just out of curiosity
            n_epoch, epoch_loss = 1, []
            for i in range(self.n_run):
                try:
                    l, _ = sess.run([self.loss, self.train_op])
                    epoch_loss.append(l)

                except tf.errors.OutOfRangeError:
                    sess.run(self.datapipe.initializer, 
                             feed_dict={self.datapipe.images_ph: self.inputs,
                                        self.datapipe.labels_ph: self.labels})

                    print('epoch: {}, loss: {}'.format(n_epoch, np.mean(epoch_loss)))
                    
                    if not(n_epoch % 5):
                        name = 'epoch_{}.ckpt'.format(n_epoch)
                        path = os.path.join(save_path, name)
                        self.saver.save(sess, path)
                        print('epoch_{} models are saved to: {}'.format(n_epoch, path))

                    # reset ops
                    n_epoch += 1
                    epoch_loss = []

            name = 'epoch_{}.ckpt'.format(n_epoch)
            path = os.path.join(save_path, name)
            self.saver.save(sess, path)
            print('final model is saved to: {}'.format(path))

    # evaluating semantic maps
    # TODO: LATER
    def evaluate(self, ckpt_path):
        with tf.Session(config=self.config) as sess:
            # init ops
            print('restoring {}...'.format(ckpt_path))
            self.saver.restore(sess, ckpt_path)
            print('restored')
            
            # datapipe initializer
            sess.run(self.datapipe.initializer, 
                     feed_dict={self.datapipe.images_ph: self.inputs,
                                self.datapipe.labels_ph: self.labels})

            in_images, in_labels, preds = sess.run([self.next_images, self.next_labels, self.preds])
        return in_images, in_labels, preds