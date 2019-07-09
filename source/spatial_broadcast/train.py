import json
import tensorflow as tf
from network import build_network
from build_graph import VAE
from load_data import DataPipeline

def train(network_specs,
          training_params,
          image_path,
          save_path):
    
    # create images DataPipeline
    datapipe = DataPipeline(image_path,
                            training_params)

    # create model VAE
    model = VAE(network_specs,
                datapipe
                training_params,
                save_path)

    # train the model
    # save_config is flexible
    model.train(save_path=save_path)

if __name__ == '__main__':
    network_specs_json = 'source/spatial_broadcast/params/original/model.json'
    training_params_json = 'source/spatial_broadcast/params/original/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    # load data
    image_folder = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

    # save config
    save_folder = 'source/spatial_broadcast/tmp/'

    train(network_specs=network_specs,
          training_params=training_params,
          image_path=image_path,
          save_folder=save_folder)