import os

import numpy as np
from tqdm import tqdm
import torch
import random
from unet_whole_model import *
import json
import pandas as pd
from PIL import Image

# Convert arrays each array of a patch dict into a tensor

def load_paths_data(folder_path: str):
    patch_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    images = []
    for patch_file in tqdm(patch_files, desc="Loading data"):
        file_name = patch_file.split(".")[0]
        _, x, y= file_name.split("_")
        
        image = {'x' : int(x), 'y' : int(y), 'patch_path' : os.path.join(folder_path,patch_file), 'augmentation' : 'no_augmentation'}
        images.append(image)
    
    images = sorted(images, key = lambda img : (img['x'], img['y']) )
    return images


def fill_test_record(test_records, modelname, model, train_data_name, other_datasets):
  new_test_record = {}
  for name in test_records.columns:
    new_test_record[name]= None

  new_test_record["model"] = [modelname]
  new_test_record["trained_on"] = [train_data_name]


  best_epoch = model.get_best_epoch()
  f,a = model.get_best_f1_accuracy()

  new_test_record['best_epoch'] = [int(best_epoch)]
  new_test_record[f"{train_data_name}_f1"] = [f]
  new_test_record[f"{train_data_name}_accuracy"] = [a]


  best_epoch_model = Model(modelname, lr = 9e-4) #lr not used here but necessary to instanciate a model
  best_epoch_model.load_model(f"{modelname}_epoch_{best_epoch}")

  for dataset in other_datasets:
    f, a = best_epoch_model.test_dataset(dataset)
    new_test_record[f"{dataset.name}_f1"]  = [f]
    new_test_record[f"{dataset.name}_accuracy"]  = [a]
  

  test_records = pd.concat([test_records,
  pd.DataFrame(new_test_record)]).reset_index(drop=True)

  test_records.to_json('test_records.jsonl', orient = 'records', lines =True)
  
  return test_records