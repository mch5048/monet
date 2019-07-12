import os
import json

import matplotlib.pyplot as plt

from model import UNet
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
    model = UNet(network_specs=network_specs,
                 datapipe=datapipe,
                 training_params=training_params,
                 mode='evaluating')

    # train the model
    # save_config is flexible
    print('''
=============
 HERE WE GO
=============
''')
    labels, preds = model.evaluate(ckpt_path=ckpt_path)

    for i in range(labels.shape[0]):
        # plt.subplots(figsize=[16,12])
        for j in range(2):
            plt.subplot(1, 4, 2*j+1)
            plt.imshow(labels[i, :, :, j])
            plt.subplot(1, 4, 2*j+2)
            plt.imshow(preds[i, :, :, j])
        plt.show()

if __name__ == '__main__':
    network_specs_json = 'source/unet/params/multid/model.json'
    training_params_json = 'source/unet/params/multid/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    # load data
    image_path = 'data/multi_dsprites_semantic_64x64_validation.npz'
    # image_path = 'data/reduced.npy'

    # save_path
    save_path = None

    # ckpt path to continue training
    ckpt_path = 'source/unet/tmp/epoch_20.ckpt'

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path,
          ckpt_path=ckpt_path)