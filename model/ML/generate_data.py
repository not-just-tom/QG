import jax
import jax.numpy as jnp
import zarr
import os 
import json

def generate_train_data(params, init_state, hr_dir):
    '''aiming for this to generate the zarr file for hr training data, and 
    also save the metadata for the run in a json file.'''

    metadata_path = os.path.join(hr_dir, "metadata.json")

    # === save metadata === #
    metadata = {
        "params_hash": params['params_hash'],
        "parameters": params,
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


    