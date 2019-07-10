'''
evaluate disentanglement
'''
import json
import numpy as np
import tensorflow as tf
from network import build_network
from build_graph import VAE
from load_data import DataPipeline
from visualize import dplot

'''
IN THE FUTURE: remove all the unnecessary print statements
'''

def evaluate(network_specs,
             ckpt_path):

    print('creating network model...')

    # create model VAE
    model = VAE(network_specs=network_specs,
                mode='evaluating')

    # train the model
    # save_config is flexible
    print('''
=============
 HERE WE GO
=============
''')

    latent_play = 10
    space = 5
    logits, preds = model.evaluate(ckpt_path=ckpt_path, 
                                   linspace=np.linspace(-2.0, 2.0, space), 
                                   latent_play=latent_play)
    print(preds.shape)
    dplot(preds, 
          space=5, 
          latent_dim=latent_play)

if __name__ == '__main__':
    network_specs_json = 'source/spatial_broadcast/params/original/model.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    # ckpt path
    ckpt_path = 'source/spatial_broadcast/tmp/epoch_80.ckpt'

    evaluate(network_specs=network_specs,
             ckpt_path=ckpt_path)