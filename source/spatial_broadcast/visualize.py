import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def dplot(images, space, latent_dim):
    n, m = latent_dim, space
    start = 1
    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(n, m, start)
        plt.imshow(hsv_to_rgb(images[i]), interpolation='nearest')
        start += 1
    plt.show()