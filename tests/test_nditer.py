'''
disentanglement test: nditer addition/subtraction
'''
import copy
import numpy as np

latent_dim = 4
z = np.random.uniform(size=[1, latent_dim])
l = np.linspace(-2.0, 2.0, 3)

z_s = []
for i in range(z.shape[1]):
    for j in np.nditer(l):
        tmp = copy.copy(z)
        tmp[0, i] += j
        z_s.append(tmp)

print(z)
z_s = np.squeeze(np.array(z_s))
print(z_s)
print(z_s.shape)