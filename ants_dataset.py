import torch
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

def preprocess_masks(mask_dir, target_size):
    """
    Preprocess the masks by resizing them and mapping grayscale values to class indices.

    Parameters:
        mask_dir (str): Path to the directory containing the masks.
        target_size (tuple): Desired size (width, height) to resize the masks.

    Returns:
        np.ndarray: Array of preprocessed masks with integer class indices.
                    Shape: (num_masks, target_size[1], target_size[0])
    """
    mask_list = []
    filenames = sorted(os.listdir(mask_dir))  # Ensure consistent order
    for file in filenames:
        mask_path = os.path.join(mask_dir, file)
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        mask = mask.resize(target_size, Image.NEAREST)  # Resize to target size
        mask_array = np.array(mask)
        
        # Map grayscale values to class indices
        unique_values = np.unique(mask_array)
        mask_normalized = np.zeros_like(mask_array)
        for idx, value in enumerate(unique_values):
            mask_normalized[mask_array == value] = idx  # Map unique values to indices
        
        mask_list.append(mask_normalized)
    return np.array(mask_list)

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

    # Ensure label has a channel dimension
    if len(label.shape) == 2:  # If label is (H, W)
        label = label.unsqueeze(0)  # Convert to (1, H, W)

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


class AntsDataset:
    def __init__(self, images_path, masks_path, target_size=(256, 256), limit=None, augmentation=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.target_size = target_size
        self.limit = limit
        self.augmentation = augmentation

        self.images = sorted([os.path.join(images_path, i) for i in os.listdir(images_path)])[:self.limit]
        self.masks = preprocess_masks(os.path.join(masks_path), self.target_size)  # Preprocess masks

        # Pre-check masks for unexpected values
        for i, mask in enumerate(self.masks):
            if 4 in np.unique(mask):  # Check if 4 exists
                self.masks[i][mask == 4] = 0

        # Verify that only valid classes exist
        for i, mask in enumerate(self.masks):
            unique_values = np.unique(mask)
            if any(value not in [0, 1, 2, 3] for value in unique_values):
                print(f"Mask {i} has unexpected values: {unique_values}")

        # Calculate mean and std during initialization
        self.mean, self.std = self.calculate_mean_std()

        # Define transformations using the calculated mean and std
        self.transform_img = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
        ])

        if self.limit is None:
            self.limit = len(self.images)

        # Initialize storage for loaded patches
        self.loaded_patches_rgb = None
        self.loaded_patches_gt = None

    def load_patches(self, training=False):
        if self.loaded_patches_rgb is None or self.loaded_patches_gt is None:
            print("Loading and transforming patches...")
            self.loaded_patches_rgb = []
            self.loaded_patches_gt = []
            
            for img_path, mask in tqdm(zip(self.images, self.masks), desc="Processing dataset", total=len(self.images)):
                # Load and transform image
                img = Image.open(img_path).convert("RGB")
                img = self.transform_img(img)
                self.loaded_patches_rgb.append(img)

                # Convert mask to tensor
                mask_tensor = torch.from_numpy(mask).long()
                self.loaded_patches_gt.append(mask_tensor)

        # Apply augmentations if required
        if self.augmentation and training:
            augmented_patches_rgb = []
            augmented_patches_gt = []

            for image, label in tqdm(zip(self.loaded_patches_rgb, self.loaded_patches_gt), desc="Applying augmentations"):
                # Generate augmentations
                new_images, new_labels, _, _ = transform_train_with_labels(image, label)
                augmented_patches_rgb.extend(new_images)  # Add augmented images
                augmented_patches_gt.extend(new_labels)  # Add augmented labels

            # Add the augmented data to the original data
            self.loaded_patches_rgb.extend(augmented_patches_rgb)
            self.loaded_patches_gt.extend(augmented_patches_gt)
        
            print(f"Total images (with augmentation): {len(self.loaded_patches_rgb)}, Total labels (with augmentation): {len(self.loaded_patches_gt)}")

    def calculate_mean_std(self):
        """
        Calculate the mean and standard deviation of the dataset.
        """
        transform = T.Compose([T.Resize(self.target_size), T.ToTensor()])
        mean = torch.zeros(3)  # For RGB images
        std = torch.zeros(3)

        print("Calculating mean and standard deviation...")
        for img_path in tqdm(self.images, desc="Processing images"):
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            mean += img_tensor.mean(dim=(1, 2))  # Mean for each channel
            std += img_tensor.std(dim=(1, 2))  # Std for each channel

        mean /= len(self.images)
        std /= len(self.images)

        return mean, std

    def __getitem__(self, index):
        # Ensure patches are loaded
        if self.loaded_patches_rgb is None or self.loaded_patches_gt is None:
            self.load_patches(training=True)  # Default to no augmentation
            #raise ValueError("Patches are not loaded. Call `load_patches` first.")

        img = self.loaded_patches_rgb[index]
        mask = self.loaded_patches_gt[index]

        return img, mask, self.images[index]

    def __len__(self):
        return min(len(self.images), self.limit)

