import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def preprocess_images(image_dir, target_size):
    """
    Preprocess the images by resizing them and normalizing pixel values.

    Parameters:
        image_dir (str): Path to the directory containing the original images.
        target_size (tuple): Desired size (width, height) to resize the images.

    Returns:
        np.ndarray: Array of preprocessed images normalized to the range [0, 1].
                    Shape: (num_images, target_size[1], target_size[0], 3)
    """
    image_list = []
    filenames = sorted(os.listdir(image_dir))  # Ensure consistent order
    for file in filenames:
        img_path = os.path.join(image_dir, file) # Loops through each file in the directory, creating the full path to the file
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
        img = img.resize(target_size, Image.ANTIALIAS)  # Resize to target size
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        image_list.append(img_array)
    return np.array(image_list)


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
        # Extract Unique Grayscale Values (ex : 0, 128, 256) and convert it to index (0, 1, 2)
        unique_values = np.unique(mask_array)
        mask_normalized = np.zeros_like(mask_array)
        for idx, value in enumerate(unique_values):
            mask_normalized[mask_array == value] = idx  # Map unique values to indices
        
        mask_list.append(mask_normalized)
    return np.array(mask_list)


# Train-Test Split
def split_data(images, masks, test_size=0.2, random_state=42):
    """
    Split the dataset into training and validation sets.

    Parameters:
        images (np.ndarray): Array of preprocessed images.
        masks (np.ndarray): Array of preprocessed masks.
        test_size (float): Proportion of the dataset to include in the validation set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Four arrays - x_train, x_val, y_train, y_val
    """
    return train_test_split(images, masks, test_size=test_size, random_state=random_state)