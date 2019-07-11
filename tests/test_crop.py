import numpy as np
import tensorflow as tf

from source.layers.ops import crop_to_fit

tf.enable_eager_execution()

down_input = np.arange(28).reshape((1, 7, 4, 1))
up_input = np.arange(10).reshape((1, 5, 2, 1))

print(np.squeeze(down_input))
print(np.squeeze(up_input))

down_input = tf.convert_to_tensor(down_input)
up_input = tf.convert_to_tensor(up_input)

test = crop_to_fit(down_input, up_input)
print(np.squeeze(test))