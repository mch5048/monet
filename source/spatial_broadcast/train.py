import json
import tensorflow as tf
from network import build_network
from build_graph import VAE
from load_data import DataPipeline

'''
IN THE FUTURE: remove all the unnecessary print statements
'''

def train(network_specs,
          training_params,
          image_path,
          save_path):
    
    print('creating datapipe...')
    # create images DataPipeline
    datapipe = DataPipeline(image_path=image_path,
                            training_params=training_params)

    print('creating network model...')
    # create model VAE
    model = VAE(network_specs=network_specs,
                datapipe=datapipe,
                training_params=training_params)

    # train the model
    # save_config is flexible
    print('''
=============
 HERE WE GO
=============
''')
    model.train(save_path=save_path)

if __name__ == '__main__':
    network_specs_json = 'source/spatial_broadcast/params/original/model.json'
    training_params_json = 'source/spatial_broadcast/params/original/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)


    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    # load data
    # image_path = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    image_path = 'data/reduced.npy'

    # save config
    save_path = 'source/spatial_broadcast/tmp/'

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_path=save_path)