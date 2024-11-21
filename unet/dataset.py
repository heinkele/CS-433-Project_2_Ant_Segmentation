import torch
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio as rio
import os
import random
from torchvision import transforms

def transform_train_with_labels(image, label):
 
    # Geometric transformation applied to both image and label
    geometric_transforms =[T.RandomVerticalFlip(p=1),
                           T.RandomHorizontalFlip(p=1),
                           T.RandomRotation(degrees = (90,90)),
                           T.RandomRotation(degrees = (180,180)),
                           T.RandomRotation(degrees = (270,270))]
    
    geometric_transforms_names = ["vertical_flip", "horizontal_flip", "rotation_90", "rotation_180°", "rotation_270"]
    
    # Randomly select 3 transformations
    selected_transforms = random.sample(list(zip(geometric_transforms, geometric_transforms_names)), 3)

    # Extract the selected transformations and their names
    geometric_transforms, geometric_transforms_names = zip(*selected_transforms)
    geometric_transforms = list(geometric_transforms)
    geometric_transforms_names = list(geometric_transforms_names)

    new_images = [transform(image) for transform in geometric_transforms]
    new_labels = [transform(label) for transform in geometric_transforms]
    
    # Other transformation applied only to the image
    if random.random() < 0.5:
      other_transforms = [T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))]
      other_transforms_name = ["gaussian_blur"]
    else:
      other_transforms = []
      other_transforms_name = []
    
    new_images = new_images +  [transform(image) for transform in other_transforms]
    new_labels = new_labels + [label for _ in other_transforms]           
    return new_images, new_labels, geometric_transforms_names , other_transforms_name

def save_to_tif(tensor, file_path):
    Image.fromarray((tensor.numpy()*255).astype(np.uint8)).save(file_path, format='TIFF')

class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict_rgb, data_dict_gt, dataset_name):
        assert len(data_dict_rgb) == len(data_dict_gt)

        self.name = dataset_name
        self.data_dict_rgb = data_dict_rgb
        self.data_dict_gt = data_dict_gt
        self.loaded_patches_rgb = None
        self.loaded_patches_gt = None
        self.means = None
        self.stds = None

    def get_tensor_image(self, image_path):
        tensor = []
        for i in range(0, len(image_path)):
            image = Image.open(image_path[i])
            img_np = np.array(image)
            tensor_image = torch.tensor(img_np)
            tensor.append(tensor_image)
        tensor_image = torch.stack(tensor)

        file_name = os.path.basename(image_path[0])
        
        if 'gt' in file_name:
            tensor_image = tensor_image.unsqueeze(-1)

        tensor_image = tensor_image.permute(0, 3, 1, 2)
        return tensor_image

    def __len__(self):
        return len(self.data_dict_rgb)


    def load_patches(self):
      if self.loaded_patches_rgb is None: 
        self.loaded_patches_rgb = [self.get_tensor_image(image_path) for image_path in tqdm(self.data_dict_rgb.values(), desc = "Loading images")]
      if self.loaded_patches_gt is None:
        self.loaded_patches_gt = [self.get_tensor_image(gt_path) for gt_path in tqdm(self.data_dict_gt.values(),  desc = "Loading groundtruths")] 
      self.means, self.stds = self.compute_means_and_stds()
    
    
    def __getitem__(self, index):
        if self.loaded_patches_rgb is None or self.loaded_patches_gt is None or index >= len(self.loaded_patches_rgb):
            raise Exception("Patches not loaded")
        return self.loaded_patches_rgb[index], self.loaded_patches_gt[index]

    def get_batch(self, index):
        if self.loaded_patches_rgb is None or self.loaded_patches_gt is None:
            raise Exception("Patches not loaded")
        return self.loaded_patches_rgb[index], self.loaded_patches_gt[index]
    
    def compute_means_and_stds(self):
        if self.loaded_patches_rgb is None:
            self.load_patches()

        batch_means = []
        batch_stds = []

        for batch_tensor in self.loaded_patches_rgb:
            batch_tensor = torch.tensor(batch_tensor).float()
            means = torch.mean(batch_tensor, axis=(0, 2, 3))
            stds = torch.std(batch_tensor, axis=(0, 2, 3))
            batch_means.append(means)
            batch_stds.append(stds)
        means_tensor = torch.stack(batch_means)
        stds_tensor = torch.stack(batch_stds)

            # Compute the overall mean and std
        means = torch.mean(means_tensor, axis=0)
        stds = torch.mean(stds_tensor, axis=0)
        return means, stds
    
    def get_images(self):
        if self.loaded_patches_rgb is None:
            raise Exception("Images not loaded")
        return self.loaded_patches_rgb
    
    def get_groundtruths(self):
        if self.loaded_patches_gt is None:
            raise Exception("Groundtruths not loaded")
        return self.loaded_patches_gt
    
    



