import tensorflow as tf
from source.monet.probability import log_gaussian

tf.enable_eager_execution()

N, H, W, C = 2, 4, 4, 3
x = tf.get_variable(name='x',
                    shape=[N, H, W, C],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer())

mu = tf.get_variable(name='mu',
                     shape=[N, H, W, 1],
                     dtype=tf.float32,
                     initializer=tf.truncated_normal_initializer())

print(x)
print(mu)
print(x - mu)

var = 0.05

log_g = log_gaussian(x=x, mu=mu, var=var)
print(log_g)