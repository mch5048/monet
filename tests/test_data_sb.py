'''
test datapipeline of dsprites
main goal: to see colors
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from source.spatial_broadcast.load_data import DataPipeline

# load data
image_folder = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

training_params = {'batch_size': 16, 'load': int(1e5)}

load = int(5e4)
datapipe = DataPipeline(image_folder, training_params, load)

print(datapipe.images[0].shape)
plt.imshow(np.squeeze(datapipe.images[0]), cmap='gray')
plt.show()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.49,
                            allow_growth=True)

config = tf.ConfigProto(gpu_options=gpu_options,
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1,
                        allow_soft_placement=True)
with tf.Session(config=config) as sess:
    sess.run(datapipe.initializer, feed_dict={datapipe.feat_ph: datapipe.images})
    images = sess.run(datapipe.next_element)
    print(images.shape)
    rgb = hsv_to_rgb(np.squeeze(images[0]))
    plt.imshow(rgb)
    plt.show()

