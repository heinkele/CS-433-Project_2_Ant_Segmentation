import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance

def augment_image_and_mask(image, mask):
    """
    Apply data augmentation transformations to an image and its corresponding mask.

    Parameters:
        image (np.ndarray): Original image with shape (height, width, channels).
        mask (np.ndarray): Corresponding segmentation mask with shape (height, width).

    Returns:
        tuple: Augmented image and mask as NumPy arrays.
    """
    # Convert NumPy arrays to PIL Images for augmentation
    img = Image.fromarray((image * 255).astype('uint8'))
    msk = Image.fromarray(mask.astype('uint8'))
    
    # Random Flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        msk = ImageOps.mirror(msk)

    if random.random() > 0.5:
        img = ImageOps.flip(img)
        msk = ImageOps.flip(msk)
    
    # Random Rotation
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        img = img.rotate(angle, resample=Image.BILINEAR)
        msk = msk.rotate(angle, resample=Image.NEAREST)

    # Random Brightness Adjustment
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)  # Adjust brightness by 80%-120%
        img = ImageOps.autocontrast(img)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)

    # Convert back to NumPy arrays
    augmented_image = np.array(img) / 255.0  # Normalize to [0, 1]
    augmented_mask = np.array(msk)

    return augmented_image, augmented_mask

def augment_dataset(images, masks, num_augmentations=2):
    """
    Generate augmented versions of the dataset.

    Parameters:
        images (np.ndarray): Array of original images. Shape: (num_images, height, width, channels).
        masks (np.ndarray): Array of original masks. Shape: (num_masks, height, width).
        num_augmentations (int): Number of augmented copies to create per original image.

    Returns:
        tuple: Augmented images and masks as NumPy arrays.
    """
    augmented_images = []
    augmented_masks = []

    for i in range(len(images)):
        # Add the original image and mask
        augmented_images.append(images[i])
        augmented_masks.append(masks[i])
        
        # Add augmented versions
        for _ in range(num_augmentations):
            augmented_image, augmented_mask = augment_image_and_mask(images[i], masks[i])
            augmented_images.append(augmented_image)
            augmented_masks.append(augmented_mask)

    return np.array(augmented_images), np.array(augmented_masks)