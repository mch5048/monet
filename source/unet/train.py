import os
import json

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
          ckpt_path,
          epoch=10):
    
    print('creating datapipe...')
    # create images DataPipeline
    datapipe = DataPipeline(image_path=image_path,
                            training_params=training_params)

    print('creating network model...')
    # create model VAE
    model = UNet(network_specs=network_specs,
                 datapipe=datapipe,
                 training_params=training_params)

    # train the model
    # save_config is flexible
    print('''
=============
 HERE WE GO
=============
''')
    model.train(save_path=save_path,
                ckpt_path=ckpt_path,
                epoch=epoch)

if __name__ == '__main__':
    network_specs_json = 'source/unet/params/multid/model.json'
    training_params_json = 'source/unet/params/multid/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    # load data
    image_path = 'data/multi_dsprites_semantic_64x64_training_3c_diff.npz'
    
    # save path
    save_path = 'source/unet/3class/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # ckpt path to continue training
    ckpt_path = 'source/unet/3class/epoch_10.ckpt'

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path,
          ckpt_path=ckpt_path,
          epoch=10)