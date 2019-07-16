import tensorflow as tf
from network import build_network
from source.misc import sampler_normal 

class VAE(object):
    def __init__(self,
                 network_spec,
                 inputs,
                 log_mask,
                 scope='component_vae',
                 reuse=False):

        self.network_spec = network_spec
        self.inputs = inputs
        self.log_mask = log_mask
        
        with tf.variable_scope(scope, reuse):
            self._build_graph()

    def _build_graph(self):
        inputs = tf.concat([self.inputs, self.log_mask], axis=-1)
        z_samples = self._build_encoder(inputs)
        self._build_decoder(z_samples)

    def _build_encoder(self, inputs):
        encoder = build_network(inputs=inputs,
                                model_specs=self.network_spec['encoder'],
                                latent_dim=self.network_spec['latent_dim'],
                                name='encoder')
        self.mean, self.log_var = tf.split(encoder, 
                                           [self.network_spec['latent_dim'], self.network_spec['latent_dim']], 
                                           axis=1)
        z_samples = sampler_normal(self.mean, 
                                   self.log_var)
        return z_samples

    def _build_decoder(self, inputs):
        logits = build_network(inputs=inputs,
                               model_specs=self.network_spec['decoder'],
                               num_channel=self.network_spec['num_channel'],
                               name='decoder')
        self.log_mask, self.image_mean = tf.split(logits,
                                                  [1, 3],
                                                  axis=-1)

    def output(self):
        return self.mean, self.log_var, self.log_mask, self.image_mean