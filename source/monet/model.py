import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

from components import VAE, UNet
from probability import log_gaussian
from source.misc import kl_divergence

class MONet(object):
    def __init__(self,
                 datapipe,
                 network_specs,
                 training_params,
                 scope='monet'):

        self.datapipe = datapipe

        self.network_specs = network_specs

        self.beta, self.gamma = 0.5, 0.5
        self.k_steps = 5

        self.lr = training_params['lr']

        with tf.variable_scope(scope):
            # losses and optimizer are built in _build_graph()
            self._build_graph()
        self.vars_initializer = tf.global_variables_initializer()

        # saver
        self.saver = tf.train.Saver()
 
        # config for session    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.69,
                                    allow_growth=True)

        self.config = tf.ConfigProto(gpu_options=gpu_options,
                                     inter_op_parallelism_threads=4,
                                     intra_op_parallelism_threads=4,
                                     allow_soft_placement=True)

    def _build_graph(self):
        logvar_bg = 2 * tf.log(0.09)
        logvar_fg = 2 * tf.log(0.11)

        batch_size = self.datapipe.batch_size
        H, W = self.datapipe.images.shape[1], self.datapipe.images.shape[2]
        log_s_k = tf.convert_to_tensor(np.zeros((batch_size, H, W, 1)).astype(np.float32))

        self.loss_RE = 0
        self.loss_KL = 0
        log_masks = []
        re_masks = []

        next_element = self.datapipe.next_images

        attention = UNet(network_specs=self.network_specs['unet'])
        component_vae = VAE(network_specs=self.network_specs['vae'])
        # summarizer = Summarizer()

        self.re_image_means = []
        epsilon = 1e-10

        for k in range(self.k_steps):
            reuse = not(k == 0)
            log_a_k = attention(images=next_element,
                                log_scope=log_s_k,
                                reuse=reuse)
            log_m_k = log_a_k + log_s_k
            log_s_k = tf.log(1.0 - tf.exp(log_a_k) + epsilon) + log_s_k

            # i forgot this step, and because of that sum_k log_m_k does not add up to 1
            if k == (self.k_steps - 1):
                log_m_k = log_s_k

            z_mean_k, z_logvar_k, re_m_k, re_image_mean_k = component_vae(images=next_element,
                                                                          log_mask=log_m_k,
                                                                          reuse=reuse)
            self.re_image_means.append(re_image_mean_k)

            logvar = logvar_bg if reuse else logvar_fg
            log_mixture = log_m_k + log_gaussian(x=next_element, mean=re_image_mean_k, logvar=logvar)
            self.loss_RE += tf.exp(log_mixture)

            self.loss_KL += kl_divergence(mu=z_mean_k, logvar=z_logvar_k)

            log_masks.append(log_m_k)
            re_masks.append(re_m_k)

        self.loss_RE = tf.reduce_sum(-tf.log(self.loss_RE + epsilon)) / batch_size
        tf.summary.scalar('loss_RE', self.loss_RE)

        self.loss_KL = self.beta * tf.reduce_sum(self.loss_KL) / batch_size
        tf.summary.scalar('loss_KL', self.loss_KL)
        
        re_masks = tf.convert_to_tensor(re_masks)
        log_masks = tf.convert_to_tensor(log_masks)

        # this is probably not correct
        self.re_log_soft_masks = tf.nn.log_softmax(re_masks, axis=0)

        # another bug here: typed re_log_masks instead of re_log_soft_masks
        loss_ATT = tf.multiply(tf.exp(log_masks), log_masks - self.re_log_soft_masks)
        
        self.loss_ATT = self.gamma * tf.reduce_sum(loss_ATT) / batch_size
        tf.summary.scalar('loss_ATT', self.loss_ATT)
        
        self.loss = self.loss_RE + self.loss_KL + self.loss_ATT
        tf.summary.scalar('loss', self.loss)
        
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

        self.merged = tf.summary.merge_all()

    def train(self, save_path, epoch=0, ckpt_path=None):
        with tf.Session(config=self.config) as sess:
            writer = tf.summary.FileWriter('source/monet/logs/test', sess.graph)
            # init ops
            if ckpt_path:
                print('restoring {}...'.format(ckpt_path))
                self.saver.restore(sess, ckpt_path)
                print('restored')
                n_run = epoch * (self.datapipe.images.shape[0] // self.datapipe.batch_size)
            else:
                sess.run(self.vars_initializer)
                n_run = self.datapipe.n_run

            # datapipe initializer
            sess.run(self.datapipe.initializer, 
                     feed_dict={self.datapipe.images_ph: self.datapipe.images})

            # n_epoch, epoch loss just out of curiosity
            n_epoch, epoch_loss = epoch + 1, []

            for i in range(n_run):
                try:
                    test, summary, l, _ = sess.run([self.re_log_soft_masks, self.merged, self.loss, self.train_op])
                    epoch_loss.append(l)
                    writer.add_summary(summary, i)
                except tf.errors.OutOfRangeError:
                    sess.run(self.datapipe.initializer, 
                             feed_dict={self.datapipe.images_ph: self.datapipe.images})

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
                     feed_dict={self.datapipe.images_ph: self.datapipe.images})

            return sess.run([self.re_image_means, self.re_log_soft_masks, self.datapipe.next_images])
