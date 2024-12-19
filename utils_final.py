import copy
import os
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm


def move_files(file_list, source_dir, target_dir):
    """
    Copies a list of files from a source directory to a target directory.

    Args:
        file_list (list of str): List of file names to be copied.
        source_dir (str): Path to the directory containing the source files.
        target_dir (str): Path to the directory where the files will be copied.

    Returns:
        None
    """
    for file_name in file_list:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(target_dir, file_name)
        shutil.copy(src_path, dst_path)  


def compute_accuracy(prediction, target):
    """
    Computes pixel-wise accuracy for multi-class segmentation.

    Args:
        prediction: Tensor of shape [B, C, H, W] .
        target: Tensor of shape [B, H, W] .

    Returns:
        Accuracy : The fraction of correctly classified pixels.
    """
    # Convert logits to class predictions
    pred_classes = torch.argmax(prediction, dim=1)  # [B, H, W]

    # Compare with ground truth
    correct = (pred_classes == target).sum().item()
    total = target.numel()

    return correct / total


def compute_f1_score(prediction, target, num_classes):
    """
    Computes the F1 score for multi-class segmentation.

    Args:
        prediction: Tensor of shape [B, C, H, W].
        target: Tensor of shape [B, H, W].
        num_classes: Number of classes in the dataset.

    Returns:
        Average F1 score across all classes.
    """
    # Convert logits to class predictions
    pred_classes = torch.argmax(prediction, dim=1)  # [B, H, W]

    # Flatten predictions and target for sklearn metrics
    pred_flat = pred_classes.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    # Compute F1 score for all classes
    f1 = f1_score(target_flat, pred_flat, labels=list(range(num_classes)), average="macro")

    return f1


def eval_model(dataloader, model, device):
    """
    Evaluates the model on the given dataloader and computes accuracy and F1 score.

    Args:
        model: Trained model to evaluate.
        dataloader: DataLoader for the test/validation set.
        device: Device to run the evaluation on (e.g., 'cpu' or 'cuda').

    Returns:
        A tuple containing:
        - Average accuracy across the dataset.
        - Average F1 score across the dataset.
    """
    model.eval()
    running_accuracy = 0
    running_f1 = 0
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].long().to(device)

            # Remove channel dimension (if mask has shape [B, 1, H, W])
            mask = mask.squeeze(1)

            # Get predictions
            y_pred = model(img)

            # Compute metrics
            accuracy = compute_accuracy(y_pred, mask)
            f1 = compute_f1_score(y_pred, mask, num_classes=y_pred.shape[1])

            # Accumulate metrics
            running_accuracy += accuracy
            running_f1 += f1

        # Calculate average metrics over the test set
        avg_accuracy = running_accuracy / (idx + 1)
        avg_f1 = running_f1 / (idx + 1)

        return avg_accuracy, avg_f1

def save_model(model, epoch, save_dir="/content/models"):
    """
    Saves the model state and training metrics for the given epoch.

    Args:
        model: The model to save.
        epoch: Current epoch number.
        train_accuracies: List of training accuracies over epochs.
        train_f1s: List of training F1 scores over epochs.
        val_accuracies: List of validation accuracies over epochs.
        val_f1s: List of validation F1 scores over epochs.
        save_dir: Directory where models and metrics will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, f"unet_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def plot_training_metrics(epochs, train_accuracies, val_accuracies, train_f1s, val_f1s):
    """
    Plots training and validation accuracy and F1 scores over epochs.

    Args:
        epochs (int): Total number of epochs.
        train_accuracies (list): List of training accuracies for each epoch.
        val_accuracies (list): List of validation accuracies for each epoch.
        train_f1s (list): List of training F1 scores for each epoch.
        val_f1s (list): List of validation F1 scores for each epoch.

    Returns:
        None
    """
    epochs_list = list(range(1, epochs + 1))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy over epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_list, val_accuracies, label='Validation Accuracy')
    plt.xticks(ticks=list(range(1, epochs + 1, 1)))
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Plot F1 Score over epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_f1s, label='Training F1 Score')
    plt.plot(epochs_list, val_f1s, label='Validation F1 Score')
    plt.xticks(ticks=list(range(1, epochs + 1, 1)))
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.show()


def mask_pred_plots(image_tensors, mask_tensors, model, device):
    """
    Visualizes predictions and ground truth masks and compute F1 Score and Accuracy for each sample.

    Args:
        image_tensors (list of Tensors): List of image tensors.
        mask_tensors (list of Tensors): List of corresponding ground truth mask tensors.
        model_pth: Path to the trained model.
        device: Device to use for computation ('cpu' or 'cuda').

    Returns:
        None
    """

    
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    num_classes = 4 

    for idx, (img_tensor, mask_tensor) in enumerate(zip(image_tensors, mask_tensors)):
        img = transform(img_tensor).to(device).unsqueeze(0).float() 
        mask = transform(mask_tensor).squeeze(0).to(device).long()  

        with torch.no_grad():
            pred_mask = model(img)  

        # Compute metrics
        f1 = compute_f1_score(pred_mask, mask, num_classes)
        acc = compute_accuracy(pred_mask, mask)

        # Visualization
        img_vis = img.squeeze(0).cpu().permute(1, 2, 0)  # Convert image for display [B, H, W] -> [H, W, B]
        pred_vis = torch.argmax(pred_mask.squeeze(0), dim=0).cpu()  # Convert logits to class indices
        mask_vis = mask.cpu()

        print(f"Sample {idx+1} - F1 Score: {f1:.4f}, Accuracy: {acc:.4f}")

        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(img_vis), plt.title("Normalized image")
        plt.subplot(132), plt.imshow(pred_vis, cmap="viridis"), plt.title("Predicted")
        plt.subplot(133), plt.imshow(mask_vis, cmap="viridis"), plt.title("Ground Truth Mask")
        plt.show()
