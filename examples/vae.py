import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

from source import layers
from source.layers import misc

# LOAD DATA
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalize
x_train, x_test = x_train.astype('float32') / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)

idx = np.arange(x_train.shape[0])

'''
# onehot encode
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
'''

initializer = tf.glorot_uniform_initializer()

# MODEL
# inference model
def encoder(x, latent_dim, name='encoder'):
    with tf.variable_scope(name):
        out = layers.conv2d(input_=x,
                            filters=32,
                            kernel_size=3,
                            stride_size=2,
                            padding='SAME',
                            activation=tf.nn.relu,
                            kernel_initializer=initializer,
                            name='conv2d_1')
        out = layers.conv2d(input_=out,
                            filters=64,
                            kernel_size=3,
                            stride_size=2,
                            padding='SAME',
                            activation=tf.nn.relu,
                            kernel_initializer=initializer,
                            name='conv2d_2')
        out = layers.flatten(out)
        out = layers.fc(input_=out,
                       units=16,
                       activation=tf.nn.relu,
                       kernel_initializer=initializer,
                       name='fc_1') 
        mu = layers.fc(input_=out,
                       units=latent_dim,
                       activation=None,
                       kernel_initializer=initializer,
                       name='mu')
        logvar = layers.fc(input_=out,
                           units=latent_dim,
                           activation=None,
                           kernel_initializer=initializer,
                           name='logvar')
    return mu, logvar

def decoder(x, name='decoder'):
    with tf.variable_scope(name):
        out = layers.fc(input_=x,
                        units=7*7*64,
                        activation=tf.nn.relu,
                        kernel_initializer=initializer,
                        name='fc_1')
        out = tf.reshape(out, shape=[-1, 7, 7, 64])

        out = layers.conv2d_transpose(input_=out,
                                      filters=64,
                                      kernel_size=3,
                                      stride_size=2,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      kernel_initializer=initializer,
                                      name='conv2d_transpose_1')
        out = layers.conv2d_transpose(input_=out,
                                      filters=32,
                                      kernel_size=3,
                                      stride_size=2,
                                      padding='SAME',
                                      activation=tf.nn.relu,
                                      kernel_initializer=initializer,
                                      name='conv2d_transpose_2')
        out = layers.conv2d_transpose(input_=out,
                                      filters=1,
                                      kernel_size=3,
                                      stride_size=1,
                                      padding='SAME',
                                      activation=None,
                                      kernel_initializer=initializer,
                                      name='conv2d_transpose_3')
        return out

# training parameters
batch_size = 100
n_epochs = 30
latent_dim = 2

# placeholders
x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='x_ph')
z_ph = tf.placeholder(dtype=tf.float32, shape=[None, latent_dim])

# computational graph
z_mu, z_logvar = encoder(x_ph, latent_dim=latent_dim)

sampler = tf.random.normal(shape=tf.shape(z_mu)) * tf.exp(z_logvar * 0.5) + z_mu

x_pred = decoder(z_ph)

# op
x_sigm = tf.nn.sigmoid(x_pred)
# 
losses
b_ph = tf.placeholder(dtype=tf.float32, shape=(), name='b_ph')

# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_pred, labels=x_ph)
# cross_entropy = tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
mse = tf.reduce_sum(tf.square(x_sigm - x_ph), axis=[1, 2, 3])

kl_divergence = 0.5 * tf.reduce_sum((tf.exp(z_logvar) + tf.square(z_mu) - z_logvar - 1.0), axis=1)
loss = tf.reduce_mean(mse + kl_divergence)

# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(loss)

b_init = 1e-2
b_end = 1.0
r = np.exp((1.0 / n_epochs) * np.log(b_end / b_init)) - 1.0
print('r: {}'.format(r))

print('\n\n-------x_train.shape----------')
print(x_train.shape)
print('-------x_train.shape----------\n\n')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(n_epochs):
        # always forget that np.random.shuffle is in-place
        np.random.shuffle(idx)
        all_batch_idx = np.array_split(idx, x_train.shape[0] // batch_size)
        epoch_loss = []
        for batch_idx in all_batch_idx:
            feed_dict = {x_ph: x_train[batch_idx], b_ph: b_init}
            z_samples = sess.run(sampler, feed_dict=feed_dict)
            feed_dict[z_ph] = z_samples
            batch_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            epoch_loss.append(batch_loss)
        b_init *= (1 + r)
        
        if e % 1 == 0:
            print('epoch: {}, mean_loss: {}'.format(e, np.mean(epoch_loss)))

    print('--------generate new MNIST--------')
    ## generate some interesting images
    n = 1
    new_samples = np.random.normal(size=[n, latent_dim])
    new_sigm = sess.run(x_sigm, feed_dict={z_ph: new_samples})

    # new_preds shape should be [10, 28, 28, 1]
    for i in range(n):
        plt.imshow(new_sigm[i, :, :, 0], cmap='gray')
        plt.show()