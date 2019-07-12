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

num_class = 3

# elements per class 
elems = 32 * 32 * 40 * 6

# this means 0:elems -> square, elems:2*elems -> ellipses, 2*elems:3*elems -> heart
# we want to have 1/3 of images to have the same class and remainig mixed
squares = images[:elems]
ellipses = images[elems:2*elems]
heart = images[2*elems:]

# number of examples to create
N = 5e4
N_same = int(N / 3)
N_mixed = int(N - N_same)

images = []
labels = []

d = {'s': 0, 'e': 0, 'h': 0, 't': 0}

def random_selector():
    # select upto 3 elements
    n_s = int(3 * np.random.uniform())
    n_e = int(3 * np.random.uniform())
    n_h = int(3 * np.random.uniform())

    idx_s = (elems * np.random.uniform(size=n_s)).astype(np.int)
    idx_e = (elems * np.random.uniform(size=n_e)).astype(np.int)
    idx_h = (elems * np.random.uniform(size=n_h)).astype(np.int)

    l_s = squares[idx_s]
    l_e = ellipses[idx_e]
    l_h = heart[idx_h]
    
    def add(x):
        x = np.sum(x, axis=0)
        x[x > 0.5] = 1.0
        return x

    add_s = add(l_s)
    add_e = add(l_e)
    add_h = add(l_h)
    
    c = np.random.uniform()
    r = 't'
    # with p=0.15 only squares, p=0.15 only ellipses
    # p=0.15 only hearts
    zeros = np.zeros((64, 64, 1))

    if c < 0.15:
        r = 's'
        imgs = add_s
        labels = np.concatenate([np.expand_dims(add_s, -1), zeros, zeros], -1)
    elif c < 0.30:
        r = 'e'
        imgs = add_e
        labels = np.concatenate([zeros, np.expand_dims(add_e, -1), zeros], -1)
    elif c < 0.45:
        r = 'h'
        imgs = add_h
        labels = np.concatenate([zeros, zeros, np.expand_dims(add_h, -1)], -1)
    else:
        imgs = add_s + add_e + add_h
        imgs[imgs > 0.5] = 1.0
        labels = np.concatenate([np.expand_dims(add_s, -1),
                                 np.expand_dims(add_e, -1),
                                 np.expand_dims(add_h, -1)], -1)
    return imgs, labels, r

'''
def random_selector(select_from1, select_from2, lb='mixed'):
    # squares
    image1_id = int(elems * np.random.uniform())
    image2_id = int(elems * np.random.uniform())

    image1, image2 = select_from1[image1_id], select_from2[image2_id]
    image = image1 + image2
    image[image > 0.5] = 1.0
    image_label = np.expand_dims(image, -1)
    zeros = np.zeros(image_label.shape)
    if lb == 'squares':
        label = np.concatenate([image_label, zeros], -1)
    elif lb == 'ellipses':
        label = np.concatenate([zeros, image_label], -1)
    else:
        label = np.concatenate([np.expand_dims(image1, -1), np.expand_dims(image2, -1)], -1)    
    return image, label

print('creating...')
for i in range(N_same):
    c = np.random.uniform()
    if c < 0.5:
        im, l = random_selector(squares, squares, lb='squares')

    else:
        # ellipses
        im, l = random_selector(ellipses, ellipses, lb='ellipses')    
    images.append(im)
    labels.append(l)

for i in range(N_mixed):
    im, l = random_selector(squares, ellipses)
    images.append(im)
    labels.append(l)
'''

for i in range(int(N)):
    im, l, r = random_selector()
    images.append(im)
    labels.append(l)
    d[r] += 1

images = np.array(images) 
labels = np.array(labels)

images = images.astype(np.float32)
labels = labels.astype(np.float32)
print(images.shape, labels.shape)

plt.subplot(411)
plt.imshow(images[20], cmap='gray')
plt.subplot(412)
plt.imshow(labels[20, :, :, 0], cmap='gray')
plt.subplot(413)
plt.imshow(labels[20, :, :, 1], cmap='gray')
plt.subplot(414)
plt.imshow(labels[20, :, :, 2], cmap='gray')
plt.show()

print('saving...')
print(d)
save_path = 'data/multi_dsprites_semantic_64x64_training_3c_diff.npz'
np.savez_compressed(save_path, imgs=images, labels=labels)
print('saved to {}'.format(save_path))