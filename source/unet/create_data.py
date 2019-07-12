'''
we need to create our own data
ellipses and squares
'''

import numpy as np
import matplotlib.pyplot as plt

'''
we will get two random images from spirites dataset
and merge them
quick and easy
'''

# load data
image_path = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

print('loading...')
with np.load(image_path) as data:
    images = data['imgs']
    classes = data['latents_classes']

num_class = 2 

# elements per class 
elems = 32 * 32 * 40 * 6

# this means 0:elems -> square, elems:2*elems -> ellipses, 2*elems:3*elems -> heart
# we want to have 1/3 of images to have the same class and remainig mixed
squares = images[0:elems]
ellipses = images[elems:2*elems]

# number of examples to create
N = 3e5
N_same = int(N / 3)
N_mixed = int(N - N_same)

images = []
labels = []

def random_selector(select_from1, select_from2):
    # squares
    image1_id = int(elems * np.random.uniform())
    image2_id = int(elems * np.random.uniform())

    image1, image2 = select_from1[image1_id], select_from2[image2_id]
    image = image1 + image2
    image[image > 0.5] = 1.0
    label = np.concatenate([np.expand_dims(image1, -1), np.expand_dims(image2, -1)], -1)
    return image, label

print('creating...')
for i in range(N_same):
    c = np.random.uniform()
    if c < 0.5:
        im, l = random_selector(squares, squares)

    else:
        # ellipses
        im, l = random_selector(ellipses, ellipses)    
    images.append(im)
    labels.append(l)

for i in range(N_mixed):
    im, l = random_selector(squares, ellipses)
    images.append(im)
    labels.append(l)

images, labels = np.array(images), np.array(labels)

print(images.shape, labels.shape)

plt.subplot(311)
plt.imshow(images[20], cmap='gray')
plt.subplot(312)
plt.imshow(labels[20, :, :, 0], cmap='gray')
plt.subplot(313)
plt.imshow(labels[20, :, :, 1], cmap='gray')
plt.show()

print('saving...')
np.savez('data/multi_dsprites_semantic_64x64.npy', imgs='images', labels='labels')
