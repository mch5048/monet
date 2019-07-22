import os
import argparse
import json

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
          logs_path,
          ckpt_path,
          sigmoid_output,
          epoch):
    
    print('creating datapipe...')
    # create images DataPipeline
    datapipe = DataPipeline(image_path=image_path,
                            training_params=training_params)

    print('creating network model...')
    # create model VAE
    model = MONet(network_specs=network_specs,
                  datapipe=datapipe,
                  training_params=training_params,
                  sigmoid_output=sigmoid_output)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--logs_path', type=str, default='tmp')    
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--sigmoid_output', type=bool, default=False)
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()

    network_specs_json = 'source/monet/params/test/model.json'
    training_params_json = 'source/monet/params/test/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    # load data
    image_path = 'data/monet_2object_colored_64x64_normalized.npz'
    
    save_path = 'source/monet/save/{}'.format(args.save_path)
    logs_path = 'source/monet/logs/{}'.format(args.logs_path)

    # ckpt path to continue training
    ckpt_path = args.ckpt_path
    epoch = args.epoch

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path,
          logs_path=logs_path,
          ckpt_path=ckpt_path,
          sigmoid_output=args.sigmoid_output,
          epoch=0)