import json
import tensorflow as tf
from network import build_network
from build_graph import VAE
from load_data import DataPipeline

def train(network_specs,
          image_size,
          latent_dim,
          training_params,
          image_path):
    
    # create images DataPipeline
    images = DataPipeline(image_path,
                          training_params)

    # create model VAE
    model = VAE(network_specs,
                sampler,
                images,
                latent_size,
                training_params)

    # train the model
    model.train()


if __name__ == '__main__':
    network_specs_json = 'source/spatial_broadcast/params/original/model.json'
    training_params_json = 'source/spatial_broadcast/params/original/params.json'

    with open(network_specs_json, 'r') as f:
        network_specs = json.load(f)

    with open(training_params_json, 'r') as f:
        training_params = json.load(f)

    image_size = [64, 64, 3]
    latent_dim = 10

    # load data
    image_folder = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
  
    train(network_specs=network_specs,
          image_size=image_size,
          latent_dim=latent_dim,
          training_params=training_params,
          image_path=image_path)