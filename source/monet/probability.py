from math import pi
import tensorflow as tf

# x = [N, H, W, C], mu = [N, H, W, 1] (should be, but check), var is scalar
def log_gaussian(x, mean, logvar):
    return -0.5 * (tf.log(2 * pi) + logvar + tf.square(x - mean) / tf.exp(logvar))
