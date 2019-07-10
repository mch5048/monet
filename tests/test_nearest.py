'''
testing cv2 nearest neighbor and my implementation for 
unet
'''

import cv2
import numpy as np

image = np.arange(16.0).reshape([4, 4])

print(image)

'''
cv2.resize(src, dst, fx, fy, interpolation)
src: source image
dst: destination image size
fx, fy: scale factors
interpolation: interpolation method
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
# cv2.INTER_NEAREST: max_pooling
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
