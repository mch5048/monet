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

        self.lr = training_params['lr']

        with tf.variable_scope(scope):
            # losses and optimizer are built in _build_graph()
            self._build_graph()
        self.vars_initializer = tf.global_variables_initializer()

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
        next_element = self.datapipe.next_images
        ###
        # step 1
        ###
        H, W = self.datapipe.images.shape[1], self.datapipe.images.shape[2]
        log_scope0 = tf.convert_to_tensor(np.zeros((16, H, W, 1)).astype(np.float32))

        attention_net1 = UNet(network_specs=self.network_specs['unet'],
                              images=next_element,
                              log_scope=log_scope0)
        log_mask1, log_scope1 = attention_net1.output()
        log_max1, log_neg1 = attention_net1.additional_output()
 
        component_vae1 = VAE(network_specs=self.network_specs['vae'],
                             images=next_element,
                             log_mask=log_mask1)
        mean_1, log_var1, log_re_mask1, re_image1 = component_vae1.output()

        ###
        # step 2
        ###
        attention_net2 = UNet(network_specs=self.network_specs['unet'],
                              images=next_element,
                              log_scope=log_scope1,
                              reuse=True)

        log_mask2, log_scope2 = attention_net2.output()
        log_max2, log_neg2 = attention_net2.additional_output()

        component_vae2 = VAE(network_specs=self.network_specs['vae'], 
                             images=next_element,
                             log_mask=log_mask2,
                             reuse=True)
        mean_2, log_var2, log_re_mask2, re_image2 = component_vae2.output()

        ###
        # step 3
        ###
        '''
        attention_net3 = UNet(network_specs=self.network_specs['unet'],
                              inputs=self.images_ph,
                              log_scope=log_scope2,
                              reuse=True)
        log_mask3, log_scope3 = attention_net3.output()
        '''

        # this is only here for clarity
        log_mask3 = log_scope2
        component_vae3 = VAE(network_specs=self.network_specs['vae'], 
                             images=next_element,
                             log_mask=log_mask3,
                             reuse=True)
        mean_3, log_var3, log_re_mask3, re_image3 = component_vae3.output()

        '''
        self.get_log_scope = [log_scope0, log_scope1, log_scope2]
        self.get_log_mask = [log_mask1, log_mask2, log_mask3]
        self.get_add = [log_max1, log_neg1, log_max2, log_neg2]
        '''

        self.mask_total = tf.exp(log_mask1) + tf.exp(log_mask2) + tf.exp(log_mask3)

        # shape check
        print('shape of reconstructed image3: ', re_image1.shape)
        print('shape of reconstructed image3: ', re_image2.shape)
        print('shape of reconstructed image3: ', re_image3.shape)

        # prepare a dataset of 2 objects
        var_bg = 0.09
        var_fg = 0.11
        ######
        ### LOSSES
        ######

        ###
        # decoder NLL given mixture density
        ###
        '''
        log_sum_mask = tf.log(tf.exp(log_mask1) + tf.exp(log_mask2) + tf.exp(log_mask3))
        log_mask1 = log_mask1 - log_sum_mask
        log_mask2 = log_mask2 - log_sum_mask
        log_mask3 = log_mask3 - log_sum_mask
        '''

        # log_mask = pixel_wise logits p of categorical distribution from attention masks 
        # re_image = pixel_wise means of a gaussian distribution
        # first loss is negative log likelihood of mixture density
        # x = [N, H, W, C], mu = [N, H, W, C] (should be, but check), var is scalar
        log_mixture1 = log_mask1 + log_gaussian(x=next_element, mu=re_image1, var=var_bg)
        log_mixture2 = log_mask2 + log_gaussian(x=next_element, mu=re_image2, var=var_fg)
        log_mixture3 = log_mask3 + log_gaussian(x=next_element, mu=re_image3, var=var_fg)

        # nll_nixture = [N, H, W, C]
        nll_mixture = tf.log(tf.exp(log_mixture1) + tf.exp(log_mixture2) + tf.exp(log_mixture3) + 1e-10)
        # reduce_nll_mixture = [N, 1]
        # not reduce_mean, reduce_sum
        self.nll_mixture = -tf.reduce_sum(nll_mixture, axis=[1, 2, 3])

        ###
        # KL divergence of factorized latent with beta
        ###

        '''
        # inv_var_K = sum(inv_var_k) -> sum(exp(-log_var)  
        # [N, l] where l is the latent dim, we can do those operations because diagonal covariance matrix
        inv_var_K = tf.exp(-log_var1) + tf.exp(-log_var2) + tf.exp(-log_var3)
        # not to allow for division by zero
        epsilon = 1e-6
        var_K = 1.0 / (inv_var_K + epsilon)
        log_var_K = -tf.log(inv_var_K)

        mean_K = var_K * ((tf.exp(-log_var1) * mean_1) + (tf.exp(-log_var2) * mean_2) + (tf.exp(-log_var3) * mean_3))
        self.kl_latent = self.beta * kl_divergence(mu=mean_K, logvar=log_var_K)
        
        '''
        
        # the above formulation is correct but unfortunately not for this.
        # try both of the formulation to see different results though
        kl1 = kl_divergence(mu=mean_1, logvar=log_var1)
        kl2 = kl_divergence(mu=mean_2, logvar=log_var2)
        kl3 = kl_divergence(mu=mean_3, logvar=log_var3)
        self.kl_latent = self.beta * (kl1 + kl2 + kl3)
        

        ###
        # attention mask loss
        ###

        # this is my interpretation of d_kl between attention masks parameterizing categorical distribution
        # produced from attention_net and recounstructed from vae

        ## might need to use concat and log_softmax
        ## this might not produce 
        log_sum = tf.log(tf.exp(log_re_mask1) + tf.exp(log_re_mask2) + tf.exp(log_re_mask3))
        log_softmax1 = log_re_mask1 - log_sum
        log_softmax2 = log_re_mask2 - log_sum
        log_softmax3 = log_re_mask3 - log_sum

        self.predictions = [log_softmax1,
                            re_image1,
                            log_softmax2,
                            re_image2,
                            log_softmax3,
                            re_image3]

        # we should create a probability distribution from log_re_mask1, log_re_mask2, log_re_mask3
        # using pixel-wise categorical distribution
        phi1 = tf.exp(log_mask1) * (log_mask1 - log_softmax1)
        phi2 = tf.exp(log_mask2) * (log_mask2 - log_softmax2)
        phi3 = tf.exp(log_mask3) * (log_mask3 - log_softmax3)

        # kl_attention = [N, 1]
        # again not reduce_mean, reduce_sum
        self.kl_attention = self.gamma * tf.reduce_sum(phi1 + phi2 + phi3, axis=[1, 2, 3])

        self.loss = tf.reduce_mean(self.nll_mixture + self.kl_latent + self.kl_attention)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, save_path, epoch=0, ckpt_path=None):
        with tf.Session(config=self.config) as sess:
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
            nll, lat, att = None, None, None

            for i in range(n_run):
                try:
                    m, p, im, nll, lat, att, l, _ = sess.run([self.mask_total, self.predictions, self.datapipe.next_images, self.nll_mixture, self.kl_latent, self.kl_attention, self.loss, self.train_op])
                    epoch_loss.append(l)
                    # print(np.mean(total, 0))
                    '''
                    for j, ls in enumerate(log_add):
                        print('log_max/neg{}: '.format(np.ceil(j/2)), np.squeeze(np.exp(ls[0])), ls.shape)
                    for j, ls in enumerate(log_s):
                        print('log_scope_{}: '.format(j), np.squeeze(ls[0]), ls.shape)
                    for j, lm in enumerate(log_m):
                        print('log_mask_{}: '.format(j+1), np.squeeze(lm[0]), lm.shape)
                    print('there is inf: ', np.argmax(r < 9e-37))
                    if not (i % 1000):
                        print(np.sum(m - 1.0))
                        print(p[0][0].shape)
                        plt.subplot(4, 3, 1)
                        plt.imshow(np.exp(p[0][0]) * p[1][0] * 255)
                        plt.subplot(4, 3, 2)
                        plt.imshow(p[1][0] * 255)
                        plt.subplot(4, 3, 3)
                        plt.imshow(np.exp(np.squeeze(p[0][0])), cmap='gray')
                        plt.subplot(4, 3, 4)
                        plt.imshow(np.exp(p[2][0]) * p[3][0] * 255)
                        plt.subplot(4, 3, 5)
                        plt.imshow(p[3][0] * 255)
                        plt.subplot(4, 3, 6)
                        plt.imshow(np.exp(np.squeeze(p[2][0])), cmap='gray')
                        plt.subplot(4, 3, 7)
                        plt.imshow(np.exp(p[4][0]) * p[5][0] * 255)
                        plt.subplot(4, 3, 8)
                        plt.imshow(p[5][0] * 255)
                        plt.subplot(4, 3, 9)
                        plt.imshow(np.exp(np.squeeze(p[4][0])), cmap='gray')
                        plt.subplot(4, 3, 11)
                        plt.imshow(im[0] * 255)
                        plt.show()
                        
                        print('nll: ', nll, 'nll.shape: ', nll.shape)
                        print('lat: ', lat, 'lat.shape: ', lat.shape)
                        print('att: ', att, 'att.shape: ', att.shape)
                        '''
                except tf.errors.OutOfRangeError:
                    sess.run(self.datapipe.initializer, 
                             feed_dict={self.datapipe.images_ph: self.datapipe.images})

                    print('epoch: {}, loss: {}'.format(n_epoch, np.mean(epoch_loss)))                        
                    print('nll: ', nll, 'nll.shape: ', nll.shape)
                    print('lat: ', lat, 'lat.shape: ', lat.shape)
                    print('att: ', att, 'att.shape: ', att.shape)
                    
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

            return sess.run(self.predictions)
