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
    re_image_mean, re_log_softmax, next_images = model.evaluate(ckpt_path=ckpt_path)
    rows = len(re_image_mean)
    cols = 3
    
    for i in range(re_image_mean[0].shape[0]):
        for k in range(rows):
            r_i, r_m = re_image_mean[k][i], re_log_softmax[k][i]
            plt.subplot(rows, cols, 3*k+1)
            plt.imshow(np.exp(r_m) * r_i * 255)
            plt.subplot(rows, cols, 3*k+2)
            plt.imshow(r_i * 255)
            plt.subplot(rows, cols, 3*k+3)
            plt.imshow(np.exp(np.squeeze(r_m)), cmap='gray')
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
    ckpt_path = 'source/monet/tmp/epoch_15.ckpt'

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path,
          ckpt_path=ckpt_path)