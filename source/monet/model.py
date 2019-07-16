from components import VAE, UNet
from probability import gaussian
from source.helpers import kl_divergence

class MONet(object):
    def __init__(self,
                 network_spec,
                 datapipe,
                 scope='unet',
                 mode='training',
                 training_params=None):
        if mode not in ['training', 'evaluating']:
            raise NotImplementedError('only training and evaluating modes are implemented')

        # network specs
        self.network_specs = network_specs
        self.vae_specs = network_specs['vae']
        self.attention_specs = network_specs['attention']

        # k_steps
        self.k_steps = network_specs['k_steps']

        # datapipe
        self.datapipe = datapipe

        # inputs and images are kind of confusing to use at the same time
        # might need to change to a unified name
        self.inputs = datapipe.images
        self.input_shape = list(self.inputs.shape[1:])
        self.labels = datapipe.labels

        self.mode = mode
        # training mode must supply training_params
        if mode == 'training':
            # training_params has all the training parameters
            self.lr = training_params['lr']
            self.loss = training_params['loss']
            self.n_run = datapipe.n_run

            with tf.variable_scope(scope):
                # losses and optimizer are built in _build_graph()
                self._build_graph()
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

    def _build_graph(self,
                     beta,
                     gamma):
        component_vae = VAE(self.vae_specs)
        attention_net = UNet(self.attention_specs)

        # for now forget about k_steps, just write it down
        # prepare a dataset of 2 objects

        re_var = 0.5

        # step 1
        log_mask1, log_scope1 = attention_net.feed(image=x, scope=log_scope0)
        mean_1, log_var1, log_re_mask1, re_image1 = component_vae.feed(image=x, mask=log_mask1)

        # step 2
        log_mask2, log_scope2 = attention_net.feed(image=x, scope=log_scope1)
        mean_2, log_var2, log_re_mask2, re_image2 = component_vae.feed(image=x, mask=log_mask2)

        # step 3
        log_mask3, log_scope3 = attention_net.feed(image=x, scope=log_scope2)
        mean_3, log_var3, log_re_mask3, re_image3 = component_vae.feed(image=x, mask=log_mask3)

        # also build loss here
        
        ###
        # decoder NLL given mixture density
        ###
        # log_mask = pixel_wise logits p of categorical distribution from attention masks 
        # re_image = pixel_wise means of a gaussian distribution
        # first loss is negative log likelihood of mixture density
        mixture1 = tf.exp(log_mask1) * gaussian(re_image1, re_var)
        mixture2 = tf.exp(log_mask2) * gaussian(re_image2, re_var)
        mixture3 = tf.exp(log_mask3) * gaussian(re_image3, re_var)

        # nll_nixture = [N, H, W, C]
        nll_mixture = -tf.log(mixture1 + mixture2 + mixture3)
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
        kl_latent = beta * kl_divergence(mu=mean_K, logvar=log_var_K)

        ###
        # attention mask loss
        ###

        # this is my interpretation of d_kl between attention masks parameterizing categorical distribution
        # produced from attention_net and recounstructed from vae
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
        kl_attention = gamma * tf.reduce_mean(phi1 + phi2 + phi3, axis=[1, 2, 3])

