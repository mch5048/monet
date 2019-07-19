import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

stdev = 0.09
var = stdev ** 2

logvar = 2 * tf.log(stdev)
# H, W = 10, 10
mean = tf.random.uniform(shape=())
x = tf.linspace(-1.0, 1.0, 1000)

def log_gaussian(x, mean, logvar):
    return -0.5 * (tf.log(2 * math.pi) + logvar + tf.square(x - mean) / tf.exp(logvar))

def gaussian(x, mean, var):
    return (1.0 / tf.sqrt(2 * math.pi * var)) * tf.exp(-0.5 * tf.square(x - mean) / var)

lg = log_gaussian(x, mean, logvar)
exp_g = tf.exp(lg)
g = gaussian(x, mean, var)

print(mean)

plt.subplot(3, 1, 1)
plt.plot(x, lg)
plt.subplot(3, 1, 2)
plt.plot(x, exp_g)
plt.subplot(3, 1, 3)
plt.plot(x, g)
plt.show()

