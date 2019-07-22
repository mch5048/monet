import os
import json
import argparse

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
          ckpt_path,
          batch_size=None):
    
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
    re_image_mean, re_log_softmax, next_images, log_masks = model.evaluate(ckpt_path=ckpt_path)
    rows = len(re_image_mean)
    cols = 5
    
    if not batch_size:
        batch_size = re_image_mean[0].shape[0]

    batch_size = min(batch_size, re_image_mean[0].shape[0])

    '''
    # log_masks shape = [5, 16, 64, 64, 1]
    for k in range(rows):
        print(np.mean(np.sum(np.exp(log_masks[:, k, :, :]), 0)))
    '''

    for i in range(batch_size):
        for k in range(rows):
            print('step {}'.format(k))
            r_i, r_m, m = re_image_mean[k][i], re_log_softmax[k][i], log_masks[k][i]
            print('re_image max, min: ', np.max(r_i), np.min(r_i))
            
            # re_image_mean comes as logits, changed it to tf.nn.sigmoid
            # they must come between 0 and 1
            '''
            for j in range(3):
                tmp = r_i[:, :, j]
                r_i[:, :, j] = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
            '''
            exp_r_m = np.exp(r_m)
            print('exp_r_m max, min: ', np.max(exp_r_m), np.min(exp_r_m))
            re_masked = np.repeat(exp_r_m, 3, 2) * r_i
            print('re_masked max, min: ', np.max(re_masked), np.min(re_masked))
            exp_m = np.exp(m)
            print('exp_m max, min: ', np.max(exp_m), np.min(exp_m), exp_m.shape)
            masked = np.repeat(exp_m, 3, 2) * r_i
            print('masked max, min: ', np.max(masked), np.min(masked))
            print('')
            print('')
            plt.subplot(rows+1, cols, cols*k+1)
            plt.imshow(re_masked)
            plt.subplot(rows+1, cols, cols*k+2)
            plt.imshow(r_i)
            plt.subplot(rows+1, cols, cols*k+3)
            plt.imshow(np.exp(np.squeeze(r_m)), cmap='gray')
            plt.subplot(rows+1, cols, cols*k+4)
            plt.imshow(masked)
            plt.subplot(rows+1, cols, cols*k+5)
            plt.imshow(np.exp(np.squeeze(m)), cmap='gray')
        plt.subplot(rows+1, cols, (rows+1)*cols-(cols//2))
        plt.imshow(next_images[i])
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=None)
    args = parser.parse_args()

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
    ckpt_path = 'source/monet/tmp/epoch_{}.ckpt'.format(args.epoch)

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path,
          ckpt_path=ckpt_path,
          batch_size=args.batch_size)