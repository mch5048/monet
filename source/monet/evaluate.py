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
          batch_size=None,
          sigmoid_output=False):
    
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

        rec_image_remask = 0
        rec_image_mask = 0
        for k in range(rows):
            # print('step {}'.format(k))
            r_i, r_m, m = re_image_mean[k][i], re_log_softmax[k][i], log_masks[k][i]
            # print('re_image max, min: ', np.max(r_i), np.min(r_i))
            
            # re_image_mean comes as logits, changed it to tf.nn.sigmoid
            # they must come between 0 and 1
            
            if not sigmoid_output:
                print('no sigmoid_output...')
                for j in range(3):
                    tmp = r_i[:, :, j]
                    r_i[:, :, j] = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
            
            exp_r_m = np.exp(r_m)
            # print('exp_r_m max, min: ', np.max(exp_r_m), np.min(exp_r_m))
            re_masked = np.repeat(exp_r_m, 3, 2) * r_i
            rec_image_remask += re_masked
            # print('re_masked max, min: ', np.max(re_masked), np.min(re_masked))
            exp_m = np.exp(m)
            # print('exp_m max, min: ', np.max(exp_m), np.min(exp_m), exp_m.shape)
            masked = np.repeat(exp_m, 3, 2) * r_i
            rec_image_mask += masked
            # print('masked max, min: ', np.max(masked), np.min(masked))
            # print('')
            # print('')
            ax1 = plt.subplot(rows+1, cols, cols*k+1)
            # ax1.set_title('slots with reconstructed masks')
            plt.imshow(re_masked)
            ax2 = plt.subplot(rows+1, cols, cols*k+2)
            # ax2.set_title('reconstructed masks from component vae')
            plt.imshow(np.exp(np.squeeze(r_m)), cmap='gray')
            ax3 = plt.subplot(rows+1, cols, cols*k+3)
            # ax3.set_title('slots with learned masks')
            plt.imshow(masked)
            ax4 = plt.subplot(rows+1, cols, cols*k+4)
            # ax4.set_title('learned masks from attention network')
            plt.imshow(np.exp(np.squeeze(m)), cmap='gray')
            ax5 = plt.subplot(rows+1, cols, cols*k+5)
            # ax5.set_title('reconstructed slots')
            plt.imshow(r_i)
            
        bottom = (rows+1)*cols-(cols//2)
        ax6 = plt.subplot(rows+1, cols, bottom - 1)
        ax6.set_title('original image')
        plt.imshow(next_images[i])
        ax7 = plt.subplot(rows+1, cols, bottom)
        ax7.set_title('with reconstructed masks')
        plt.imshow(rec_image_remask)
        ax8 = plt.subplot(rows+1, cols, bottom + 1)
        ax8.set_title('with learned masks')
        plt.imshow(rec_image_mask)
        
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--sigmoid_output', action='store_true')    
    args = parser.parse_args()

    network_specs_json = 'source/monet/params/test/model.json'
    training_params_json = 'source/monet/params/test/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    # load data
    image_path = 'data/monet_2object_colored_64x64_normalized_validation.npz'
    # image_path = 'data/reduced.npy'

    # save_path
    save_path = None

    # ckpt path to continue training or to evaluate
    ckpt_path = args.ckpt_path
    batch_size = args.batch_size
    sigmoid_output = args.sigmoid_output

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path,
          ckpt_path=ckpt_path,
          batch_size=batch_size,
          sigmoid_output=sigmoid_output)