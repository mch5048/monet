from math import pi
import tensorflow as tf

# x = [N, H, W, C], mu = [N, H, W, 1] (should be, but check), var is scalar
def log_gaussian(x, mu, var):
    return -0.5 * (tf.log(2 * pi * var) + tf.square(x - mu) / var)