import tensorflow as tf

# compute KL(q(z|X) || p(z)) where p(z) comes from N(0, I)
def kl_divergence(mu, logvar):
    kl = 0.5 * (tf.exp(logvar) + tf.square(mu) - logvar - 1.0)
    return tf.reduce_mean(tf.reduce_sum(kl, axis=1))

# compute binary cross entropy
def cross_entropy(logits, labels):
    batch_size = tf.shape(logits)[0]
    logits = tf.reshape(logits, shape=[batch_size, -1])
    labels = tf.reshape(labels, shape=[batch_size, -1])
    ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(tf.reduce_sum(ce, axis=1))

# sampling from normal distribution
def sampler_normal(mu, logvar):
    return tf.random.normal(shape=tf.shape(logvar)) * tf.exp(logvar * 0.5) + mu

# spatial broadcast operator
def spatial_broadcast(z, w, h, name='spatial_broadcast'):
    with tf.name_scope(name):
        # look at dynamic vs static shape
        batch_size = tf.shape(z)[0]
        k = z.get_shape()[-1]

        z_b = tf.tile(input=z, multiples=[1, h * w], name='tile')
        z_b = tf.reshape(z_b, shape=[batch_size, h, w, k])

        print(z_b.get_shape())

        '''
        indexing does NOT matter
        1. square output (because we have square images) 2. channels are order invariant
        '''
        x = tf.linspace(-1.0, 1.0, num=w)
        y = tf.linspace(-1.0, 1.0, num=w)
        x_b, y_b = tf.meshgrid(x, y)
        
        # we need (w, w, 1)
        x_b = tf.expand_dims(x_b, axis=-1)
        y_b = tf.expand_dims(y_b, axis=-1)

        # apply concat to each sample in z_b
        z_sb = tf.map_fn(fn=lambda z_i: tf.concat([z_i, x_b, y_b], axis=-1), 
                         elems=z_b,
                         parallel_iterations=True,
                         back_prop=True,
                         swap_memory=True,
                         name='map_concat')
    return z_sb