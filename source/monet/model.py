from components import VAE, UNet
from probability import log_gaussian
from source.misc import kl_divergence

class MONet(object):
    def __init__(self,
                 datapipe,
                 network_specs,
                 scope='monet'):

        self.datapipe = datapipe

        self.network_specs = network_specs

        self.beta, self.gamma = 0.5, 0.5

        with tf.variable_scope(scope):
            # losses and optimizer are built in _build_graph()
            self._build_placeholders()
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
        next_element = self.datapipe.next_element
        ###
        # step 1
        ###
        # define log_scope0
        log_scope0 = ###TODO

        attention_net1 = UNet(network_specs=self.network_specs['unet'],
                              inputs=next_element,
                              log_scope=log_scope0)
        log_mask1, log_scope1 = attention_net1.output()

        component_vae1 = VAE(network_specs=self.network_specs['vae'],
                             inputs=next_element,
                             log_mask=log_mask1)
        mean_1, log_var1, log_re_mask1, re_image1 = component_vae1.output()

        ###
        # step 2
        ###
        attention_net2 = UNet(network_specs=self.network_specs['unet'],
                              inputs=next_element,
                              log_scope=log_scope1,
                              reuse=True)

        log_mask2, log_scope2 = attention_net2.output()

        component_vae2 = VAE(network_specs=self.network_specs['vae'], 
                             inputs=next_element,
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
                             inputs=next_element,
                             log_mask=log_mask3,
                             reuse=True)
        mean_3, log_var3, log_re_mask3, re_image3 = component_vae3.output()

        self.predictions = [log_mask1,
                            re_image1,
                            log_mask2,
                            re_image2,
                            log_mask3,
                            re_image3]

        # shape check
        print('shape of reconstructed image3: ', re_image3.shape)

        # prepare a dataset of 2 objects
        var_bg = 0.09 * 0.09
        var_fg = 0.11 * 0.11
        ######
        ### LOSSES
        ######

        ###
        # decoder NLL given mixture density
        ###
        # log_mask = pixel_wise logits p of categorical distribution from attention masks 
        # re_image = pixel_wise means of a gaussian distribution
        # first loss is negative log likelihood of mixture density
        # x = [N, H, W, C], mu = [N, H, W, 1] (should be, but check), var is scalar
        log_mixture1 = log_mask1 + log_gaussian(x=next_element, mu=re_image1, var=var_bg)
        log_mixture2 = log_mask2 + log_gaussian(x=next_element, mu=re_image2, var=var_fg)
        log_mixture3 = log_mask3 + log_gaussian(x=next_element, mu=re_image3, var=var_fg)

        # nll_nixture = [N, H, W, C]
        nll_mixture = -tf.log(tf.exp(mixture1) + tf.exp(mixture2) + tf.exp(mixture3))
        # reduce_nll_mixture = [N, 1]
        reduce_nll_mixture = tf.reduce_mean(nll_mixture, axis=[1, 2, 3])

        ###
        # KL divergence of factorized latent with beta
        ###

        # inv_var_K = sum(inv_var_k) -> sum(exp(-log_var)  
        # [N, l] where l is the latent dim, we can do those operations because diagonal covariance matrix
        inv_var_K = tf.exp(-log_var1) + tf.exp(-log_var2) + tf.exp(-log_var3)
        # not to allow for division by zero
        epsilon = 1e-6
        var_K = 1.0 / (inv_var_K + epsilon)
        log_var_K = -log(inv_var_K)

        mean_K = var_K * ((tf.exp(-log_var1) * mean_1) + (tf.exp(-log_var2) * mean_2) + (tf.exp(-log_var3) * mean_3))

        # change this beta appropriately
        kl_latent = self.beta * kl_divergence(mu=mean_K, logvar=log_var_K)

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

        # we should create a probability distribution from log_re_mask1, log_re_mask2, log_re_mask3
        # using pixel-wise categorical distribution
        phi1 = tf.exp(log_mask1) * (log_mask1 - log_softmax1)
        phi2 = tf.exp(log_mask2) * (log_mask2 - log_softmax2)
        phi3 = tf.exp(log_mask3) * (log_mask3 - log_softmax3)

        # kl_attention = [N, 1]
        kl_attention = self.gamma * tf.reduce_mean(phi1 + phi2 + phi3, axis=[1, 2, 3])

        self.loss = reduce_nll_mixture + kl_latent + kl_attention

        optimizer = tf.train.AdamOptimizer(lr=1e-4)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, save_path, epoch=0, ckpt_path=None):
        with tf.Session(config=self.config) as sess:
            # init ops
            if ckpt_path:
                print('restoring {}...'.format(ckpt_path))
                self.saver.restore(sess, ckpt_path)
                print('restored')
                self.n_run = epoch * (self.inputs.shape[0] // self.datapipe.batch_size)
            else:
                sess.run(self.vars_initializer)

            # datapipe initializer
            sess.run(self.datapipe.initializer, 
                     feed_dict={self.datapipe.images_ph: self.datapipe.images})

            # n_epoch, epoch loss just out of curiosity
            n_epoch, epoch_loss = epoch + 1, []
            for i in range(self.n_run):
                try:
                    l, _ = sess.run([self.loss, self.train_op])
                    epoch_loss.append(l)

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

            return sess.run(self.predictions)
