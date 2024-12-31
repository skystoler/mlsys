import pickle
import torch
from utils.data import PICKLE_PATH

def load_input_from_pickle_to_device(device):
    images_t, masks_t, token_ids_t = simple_load_input_from_pickle()
    images_t = images_t.to(device)
    masks_t = masks_t.to(device)
    token_ids_t = token_ids_t.to(device)
    return images_t, masks_t, token_ids_t

def simple_load_input_from_pickle():
    with open(PICKLE_PATH, 'rb') as pickle_file:
        data_dict = pickle.load(pickle_file)
    images, masks, token_ids = data_dict['stacked_images'], data_dict['stacked_masks'], data_dict['stacked_token_ids']
    images_t = images.half()
    masks_t = masks.half()
    token_ids_t = token_ids.to(torch.int64)
    return images_t, masks_t, token_ids_t