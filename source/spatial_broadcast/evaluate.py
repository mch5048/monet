'''
evaluate disentanglement
'''
import json
import tensorflow as tf
from network import build_network
from build_graph import VAE
from load_data import DataPipeline
from visualize import DPlot

'''
IN THE FUTURE: remove all the unnecessary print statements
'''

def evaluate(network_specs,
             ckpt_path,
             scope):

    print('creating network model...')

    # create model VAE
    model = VAE(network_specs=network_specs,
                mode='evaluating',
                scope=scope)

    # train the model
    # save_config is flexible
    print('''
=============
 HERE WE GO
=============
''')

    logits, preds = model.evaluate(ckpt_path=ckpt_path, linspace=np.linspace(-2.0, 2.0, 10))
    DPlot.plot(preds)

if __name__ == '__main__':
    network_specs_json = 'source/spatial_broadcast/params/original/model.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    # ckpt path
    ckpt_path = 'source/spatial_broadcast/tmp/final.ckpt'

    # scope to restore variables
    scope = 'vae/decoder'

    train(network_specs=network_specs,
          ckpt_path=ckpt_path,
          scope=scope)