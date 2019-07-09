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