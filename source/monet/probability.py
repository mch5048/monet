from math import pi
import tensorflow as tf

# x = [N, H, W, C], mu = [N, H, W, 1] (should be, but check), var is scalar
def log_gaussian(x, mu, logvar):
    return -0.5 * logvar - 0.5 * tf.square(x - mu) / tf.exp(logvar)