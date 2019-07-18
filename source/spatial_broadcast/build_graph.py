import os
import copy

import numpy as np
import tensorflow as tf

from source import misc
from network import build_network

class VAE(object):
    def __init__(self,
                 network_specs,
                 mode='training',
                 scope='vae',
                 datapipe=None,
                 training_params=None):

        # network specs
        self.network_specs = network_specs
        self.latent_dim = network_specs['latent_dim']
        self.num_channel = 3

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
            with tf.variable_scope(scope):
                self.latent_ph = tf.placeholder(dtype=tf.float32, 
                                                shape=[None, self.latent_dim], 
                                                name='latent_dim')
                self._build_decoder(self.latent_ph)

        # saver
        self.saver = tf.train.Saver()
 
        # config for session    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49,
                                    allow_growth=True)

        self.config = tf.ConfigProto(gpu_options=gpu_options,
                                     inter_op_parallelism_threads=4,
                                     intra_op_parallelism_threads=4,
                                     allow_soft_placement=True)


    def _build_graph(self):
        encoder = build_network(inputs=self.datapipe.next_element,
                                model_specs=self.network_specs['encoder'],
                                latent_dim=self.latent_dim,
                                name='encoder')
        self.mu, self.logvar = tf.split(encoder, [self.latent_dim, self.latent_dim], axis=1)
        self.z_samples = misc.sampler_normal(self.mu, self.logvar)
        self._build_decoder(self.z_samples)

    def _build_decoder(self, inputs):
        self.logits = build_network(inputs=inputs,
                                    model_specs=self.network_specs['decoder'],
                                    num_channel=self.num_channel,
                                    name='decoder')
        self.preds = tf.nn.sigmoid(self.logits)

    def _build_loss(self):
        if self.loss == 'cross_entropy':
            rec_loss = misc.cross_entropy(logits=self.logits, labels=self.datapipe.next_element)
        elif self.loss == 'mse':
            rec_loss = misc.mse(preds=preds, labels=self.datapipe.next_element)
        else:
            raise NotImplementError('loss: {} is not implemented.'.format(self.loss))

        self.kl_loss = misc.kl_divergence(mu=self.mu, logvar=self.logvar)
        self.loss = rec_loss + self.kl_loss

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
                     feed_dict={self.datapipe.feat_ph: self.inputs})

            # n_epoch, epoch loss just out of curiosity
            n_epoch, epoch_loss = 1, []
            for i in range(self.n_run):
                try:
                    k, l, _ = sess.run([self.kl_loss, self.loss, self.train_op])
                    epoch_loss.append(l)
                    if not (i % 1000):
                        print('loss: ', l, l.shape)
                        print('kl loss: ', k, k.shape)
                except tf.errors.OutOfRangeError:
                    sess.run(self.datapipe.initializer, 
                             feed_dict={self.datapipe.feat_ph: self.inputs})

                    print('epoch: {}, loss: {}'.format(n_epoch, np.mean(epoch_loss)))
                    
                    if not(n_epoch % 20):
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

    # evaluating disentanglement
    def evaluate(self, ckpt_path, linspace, latent_play):
        with tf.Session(config=self.config) as sess:
            # init ops
            print('restoring {}...'.format(ckpt_path))
            self.saver.restore(sess, ckpt_path)
            print('restored')

            latent = np.random.normal(size=[1, self.latent_dim])
            
            # crude solution, i will fix this later
            d_latent = []
            print('creating disentangled pertubrations...')
            for i in range(latent_play):
                for j in np.nditer(linspace):
                    tmp = copy.copy(latent)
                    tmp[0, i] += j
                    d_latent.append(tmp)
            d_latent = np.squeeze(np.array(d_latent))
            print('done creating!!!')

            logits, preds = sess.run([self.logits, self.preds],
                                     feed_dict={self.latent_ph: d_latent})

        return logits, preds