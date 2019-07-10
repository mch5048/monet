import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class DataPipeline(object):
    def __init__(self, image_path, training_params):
        
        try:
            with np.load(image_path) as data:
                images = data['imgs']
        except:
            # minor hacks for speedy loading
            # changed from .npz to .npy with reduced size
            images = np.load(image_path)
        
        print(images.shape)
        images = np.expand_dims(images, axis=-1)

        # tf.cast() ???
        images = images.astype(np.float32)
        # complete shuffle
        idx = np.arange(images.shape[0])
        
        print('begin: complete shuffling...')
        np.random.shuffle(idx)
        images = images[idx]

        load = int(training_params['load'])
        self.images = images[0:load]
        print(self.images.shape)
        print('end: complete shuffling...')
        
        # hard-coded, not good
        self.num_channel = 1
        self.batch_size = training_params['batch_size']
        self.color_method = training_params['color']

        # simple function dict
        self.func = {'hsv': self._hsv}

        with tf.name_scope('pipeline'):
            self._build_dataset()

        # params.json must contain either iterations or epochs
        try:
            self.n_run = int(training_params['iterations'])
        except KeyError:
            self.n_run = int(training_params['epochs']) * (self.images.shape[0] // self.batch_size)

    # we will use HSV to add colors
    # assuming image is [h, w, 1]
    def _hsv(self, image):
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
        dataset = dataset.shuffle(self.images.shape[0] // 100)
        if self.color_method:
            dataset = dataset.map(self.func[self.color_method], 
                                  num_parallel_calls=4)
            self.num_channel = 3
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=1)
        iterator = dataset.make_initializable_iterator()
        self.initializer = iterator.initializer
        self.next_element = iterator.get_next()