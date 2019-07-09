import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DataPipeline(object):
    def __init__(self, image_path, training_params):
        with np.load(image_path) as data:
            images = data['imgs']
        images = np.expand_dims(images, axis=-1)

        # tf.cast() ???
        images = images.astype(np.float32)
        # complete shuffle
        idx = images.shape[0]

        print('begin: complete shuffling...')
        np.random.shuffle(idx)
        self.images = images[idx]
        print('end: complete shuffling...')
        
        self.batch_size = training_params['batch_size']
        
        with tf.name_scope('pipeline'):
            self._build_dataset()

    # we will use HSV to add colors
    # assuming image is [h, w, 1]
    def _add_colors(image):
        h = tf.random.uniform(shape=(), 
                              minval=0.0, 
                              maxval=1.0,
                              name='h')
        s = tf.random.uniform(shape=(), 
                              minval=0.3, 
                              maxval=0.7,
                              name='s')
        v = tf.random.uniform(shape=(), 
                              minval=0.3, 
                              maxval=0.7,
                              name='v')
        # using operator overloading
        h_c = h * image
        s_c = s * image
        v_c = v * image

        return tf.concat([h_c, s_c, v_c], axis=-1)
        
    def _build_dataset(self):
        self.feat_ph = tf.placeholder(self.images.dtype, self.images.shape)
        dataset = tf.data.Dataset.from_tensor_slices(self.feat_ph)
        dataset = dataset.shuffle(self.images.shape // 100)
        dataset = dataset.map(self._add_colors, num_parallel_calls=8)
        iterator = dataset.make_initializable_iterator()
        self.initializer = iterator.initializer
        self.next_element = iterator.get_next()