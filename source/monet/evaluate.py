import os
import json

import numpy as np
import matplotlib.pyplot as plt

from model import MONet
from load_data import DataPipeline

import sys

'''
IN THE FUTURE: remove all the unnecessary print statements
'''

def train(network_specs,
          training_params,
          image_path,
          save_path,
          ckpt_path):
    
    print('creating datapipe...')
    # create images DataPipeline
    datapipe = DataPipeline(image_path=image_path,
                            training_params=training_params)

    print('creating network model...')
    # create model VAE
    model = MONet(network_specs=network_specs,
                  datapipe=datapipe,
                  training_params=training_params)

    # train the model
    # save_config is flexible
    print('''
=============
 HERE WE GO
=============
''')
    p = model.evaluate(ckpt_path=ckpt_path)
    for i in range(p[0].shape[0]):
        plt.subplot(4, 3, 1)
        plt.imshow(np.exp(p[0][i]) * p[1][i] * 255)
        plt.subplot(4, 3, 2)
        plt.imshow(p[1][i] * 255)
        plt.subplot(4, 3, 3)
        plt.imshow(np.exp(np.squeeze(p[0][i])), cmap='gray')
        plt.subplot(4, 3, 4)
        plt.imshow(np.exp(p[2][i]) * p[3][i] * 255)
        plt.subplot(4, 3, 5)
        plt.imshow(p[3][i] * 255)
        plt.subplot(4, 3, 6)
        plt.imshow(np.exp(np.squeeze(p[2][i])), cmap='gray')
        plt.subplot(4, 3, 7)
        plt.imshow(np.exp(p[4][i]) * p[5][i] * 255)
        plt.subplot(4, 3, 8)
        plt.imshow(p[5][i] * 255)
        plt.subplot(4, 3, 9)
        plt.imshow(np.exp(np.squeeze(p[4][i])), cmap='gray')
        plt.subplot(4, 3, 11)
        plt.imshow(p[6][i] * 255)
        plt.show()

if __name__ == '__main__':
    network_specs_json = 'source/monet/params/test/model.json'
    training_params_json = 'source/monet/params/test/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    # load data
    image_path = 'data/monet_2object_colored_64x64_normalized.npz'
    # image_path = 'data/reduced.npy'

    # save_path
    save_path = None

    # ckpt path to continue training or to evaluate
    ckpt_path = 'source/monet/tmp/epoch_30.ckpt'

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path,
          ckpt_path=ckpt_path)