'''
testing cv2 nearest neighbor and my implementation for 
unet
'''

import cv2
import numpy as np
import tensorflow as tf
from source.layers.ops import upsampling_2d

tf.enable_eager_execution()

image = np.arange(16.0).reshape([4, 4])

print(image)

'''
cv2.resize(src, dst, fx, fy, interpolation)
src: source image
dst: destination image size
fx, fy: scale factors
interpolation: interpolation method
'''

'''
## downsampling
# cv2.INTER_NEAREST: max_pooling
dst = cv2.resize(image, 
                 None, 
                 fx=0.5, 
                 fy=0.5, 
                 interpolation=cv2.INTER_NEAREST)

print(dst)

# cv2.INTER_LINEAR: average_pooling
dst = cv2.resize(image, 
                 None, 
                 fx=0.5, 
                 fy=0.5, 
                 interpolation=cv2.INTER_LINEAR)

print(dst)

## upsampling
dst = cv2.resize(image, 
                 None, 
                 fx=2.0, 
                 fy=2.0, 
                 interpolation=cv2.INTER_NEAREST)

print(dst)

# cv2.INTER_LINEAR: average_pooling
dst = cv2.resize(image, 
                 None, 
                 fx=2.0, 
                 fy=2.0, 
                 interpolation=cv2.INTER_LINEAR)

print(dst)
'''

image = np.transpose(np.transpose(np.tile(image, 4)).reshape([4, 4, 4]), axes=(0, 2, 1))
image = np.expand_dims(image, -1)
upsampled = upsampling_2d(tf.convert_to_tensor(image), 2)
print(np.squeeze(image[0]))
print(np.squeeze(upsampled[0]))