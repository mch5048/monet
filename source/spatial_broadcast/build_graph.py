import tensorflow as tf
import numpy as np

from source import misc
from network import build_network

class VAE(object):
    def __init__(self,
                 network_specs,
                 images,
                 latent_dim,
                 training_params,
                 scope='vae'):

        # network specs
        self.network_specs = network_specs

        # size
        self.images = images
        self.latent_dim = latent_dim

        # training_params has all the training parameters
        self.lr = training_params['lr']

        with tf.variable_scope(scope):
            self._build_placeholders()
            self._build_graph()
            self._build_loss()
            self._build_optimizer()

    # this placeholder structure WILL BE CHANGED
    # we need to build an input/output pipeline
    def _build_placeholders(self):
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None] + self.images.size, name='input_ph')
        # self.latent_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.latent_size], name='latent_size')

    def _build_graph(self):
        self.mu, self.logvar = build_network(inputs=self.input_ph,
                                             model_specs=self.network_specs['encoder'],
                                             latent_dim=self.latent_dim,
                                             name='encoder')
        self.z_samples = misc.sampler_normal(self.mu, self.logvar)
        self.logits = build_network(inputs=self.z_samples,
                                    model_specs=self.network_specs['decoder'],
                                    num_channel=self.image_size[-1],
                                    name='decoder')

    def _build_loss(self):
        ce_loss = misc.cross_entropy(logits=self.logits, labels=self.input_ph)
        kl_loss = misc.kl_divergence(mu=self.mu, logvar=self.logvar)
        self.loss = ce_loss + kl_loss

    def _build_optimizer(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, images):
        n_epochs = self.training_params['n_epochs']
        
        # we are doing manual batching here
        # we WILL BUILD an input/output pipeline
        batch_size = self.batch_size

        N = images.shape[0]
        idx = np.arange(N)

        for e in range(n_epochs):
            np.random.shuffle(idx)
            batch_idx = np.array_split(idx, N // batch_size)
            epoch_loss = []
            for b in batch_idx:
                l, _ = sess.run([self.loss, self.train_op], feed_dict={self.input_ph: images[b]})
                epoch_loss.append(l)

            if not(e % 10):
                print('epoch: {}, loss: {}'.format(e, np.mean(epoch_loss)))
