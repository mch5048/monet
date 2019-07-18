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
N = 5e4
N_same = int(N / 3)
N_mixed = int(N - N_same)

def random_selector():
    idx_s = int(elems * np.random.uniform())
    idx_e = int(elems * np.random.uniform())

    l_s = squares[idx_s]
    l_e = ellipses[idx_e]
    l_s = np.expand_dims(l_s, -1)
    l_e = np.expand_dims(l_e, -1)

    # print('l_s shape: ', l_s.shape)
    s_idx = np.nonzero(l_s)
    e_idx = np.nonzero(l_e)
    
    tmp = l_s + l_e
    # print('tmp shape:', tmp.shape)

    tmp[tmp > 0.5] = 1.0
    b_idx = np.nonzero(1.0 - tmp)

    # print('background transpose: ', np.transpose(b_idx).shape)
    background = np.ones((64, 64, 1))
    background[b_idx] = 0.0
    # print('background is ready')

    front = np.random.uniform()
    if front < 0.5:
        # square is on the front
        l_e[s_idx] = 0.0
    else:
        # ellipse is on the front
        l_s[e_idx] = 0.0

    rl_s = np.repeat(l_s, 3, axis=2)
    rl_e = np.repeat(l_e, 3, axis=2)
    background_colored = np.repeat(background, 3, axis=2)

    # random coloring
    c_s = np.random.uniform(size=3)
    c_e = np.random.uniform(size=3)
    c_b = np.random.uniform(size=3)

    rl_s = rl_s * c_s
    rl_e = rl_e * c_e
    background_colored = background_colored * c_b

    c = np.random.uniform()
    r = 't'
    # with p=0.10 only one square, p=0.10 only one ellipse
    zeros = np.zeros((64, 64, 1))

    epsilon = 1e-10
    log_zeros = np.log(zeros + epsilon)
    log_background = np.log(background + epsilon)
    log_square = np.log(l_s + epsilon)
    log_ellipse = np.log(l_e + epsilon)
    if c < 0.10:
        r = 's'
        imgs = rl_s + background_colored
        masks = np.concatenate([log_background, log_square, log_zeros], axis=-1)
    elif c < 0.20:
        r = 'e'
        imgs = rl_e + background_colored
        masks = np.concatenate([log_background, log_zeros, log_ellipse], axis=-1)
    else:
        imgs = rl_s + rl_e + background_colored
        masks = np.concatenate([log_background, log_square, log_ellipse], axis=-1)
    return imgs, masks, r

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


images = []
masks = []

d = {'s': 0, 'e': 0, 't': 0}


for i in range(int(N)):
    im, m, r = random_selector()
    images.append(im)
    masks.append(m)
    d[r] += 1

images = np.array(images) 
masks = np.array(masks)

images = images.astype(np.float32)
masks = masks.astype(np.float32)
print(images.shape)
print(masks.shape)

def plot(image, masks):
    mask1 = np.exp(masks[:, :, 0])
    mask2 = np.exp(masks[:, :, 1])
    mask3 = np.exp(masks[:, :, 2])
    plt.subfigure(4, 2, 1)
    plt.imshow(mask1, cmap='gray')
    plt.subfigure(4, 2, 2)
    plt.imshow(mask1 * image * 255)

    plt.subfigure(4, 2, 3)
    plt.imshow(mask2, cmap='gray')
    plt.subfigure(4, 2, 4)
    plt.imshow(mask2 * image * 255)

    plt.subfigure(4, 2, 5)
    plt.imshow(mask2, cmap='gray')
    plt.subfigure(4, 2, 6)
    plt.imshow(mask2 * image * 255)

    plt.subfigure(4, 2, 7)
    plt.imshow(image * 255)
    plt.show()

plot(images[20], masks[20])
plot(images[30000], masks[30000])
plot(images[400], masks[400])

print('saving...')
print(d)
save_path = 'data/monet_2object_colored_64x64_normalized_w_masks.npz'
np.savez_compressed(save_path, imgs=images, masks=masks)
print('saved to {}'.format(save_path))