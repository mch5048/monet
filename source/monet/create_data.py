'''
we need to create our own data
ellipses and squares
'''

import numpy as np
import matplotlib.pyplot as plt

'''
we will get two random images from spirites dataset
and color them and merge them
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
N = 2e4
N_same = int(N / 3)
N_mixed = int(N - N_same)

images = []

d = {'s': 0, 'e': 0, 't': 0}

def random_selector():
    idx_s = int(elems * np.random.uniform())
    idx_e = int(elems * np.random.uniform())

    l_s = squares[idx_s]
    l_e = ellipses[idx_e]
    l_s = np.expand_dims(l_s, -1)
    l_e = np.expand_dims(l_e, -1)
    
    tmp = l_s + l_e
    # print('tmp shape:', tmp.shape)

    tmp[tmp > 0.5] = 1.0

    # print('background transpose: ', np.transpose(b_idx).shape)
    background = np.ones((64, 64, 1))
    background[tmp > 0.5] = 0.0

    front = np.random.uniform()
    if front < 0.5:
        # square is on the front
        l_e[l_s > 0.5] = 0.0
    else:
        # ellipse is on the front
        l_s[l_e > 0.5] = 0.0

    l_s = np.repeat(l_s, 3, axis=2)
    l_e = np.repeat(l_e, 3, axis=2)
    background = np.repeat(background, 3, axis=2)

    # random coloring
    c_s = np.random.uniform(size=3)
    c_e = np.random.uniform(size=3)
    c_b = np.random.uniform(size=3)

    l_s = l_s * c_s
    l_e = l_e * c_e
    background = background * c_b

    c = np.random.uniform()
    r = 't'
    # with p=0.10 only one square, p=0.10 only one ellipse

    '''
    if c < 0.10:
        r = 's'
        imgs = l_s + background
    elif c < 0.20:
        r = 'e'
        imgs = l_e + background
    else:
        imgs = l_s + l_e + background
    '''
    imgs = l_s + l_e + background
    print(np.max(imgs), np.max(l_s), np.max(l_e), np.max(background))
    return imgs, r

for i in range(int(N)):
    im, r = random_selector()
    images.append(im)
    d[r] += 1

images = np.array(images) 

images = images.astype(np.float32)
print(images.shape, np.max(images), np.min(images))

idx = [i for i in range(10)]
for i in idx:
    plt.imshow(images[i] * 255.0)
    plt.show()

print('saving...')
print(d)
save_path = 'data/monet_2object_colored_64x64_normalized.npz'
np.savez_compressed(save_path, imgs=images)
print('saved to {}'.format(save_path))