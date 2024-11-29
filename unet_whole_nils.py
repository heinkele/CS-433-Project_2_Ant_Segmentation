import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from dataset import *
from utils_nils import *
import json
from unet_model_nils import UNet
from torcheval.metrics import BinaryF1Score as F1_score
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os

class Model:
    def __init__(self, model_name, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = UNet(n_channels=3, n_classes=1)  # Assuming RGB input
        self.unet.to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.threshold = 0.5  # Adjust threshold for binary predictions
        self.model_name = model_name
        self.f1 = F1_score(threshold=self.threshold, device=self.device)
        self.batch_size = 16
        self.all_histories = {
            "Train": {"Train loss": [], "F1_score": [], "Accuracy": []},
            "Validation": {"Train loss": [], "F1_score": [], "Accuracy": []},
        }
        self.model_path = f"./{self.model_name}/"

        self.means = None
        self.stds = None
        self.transform = None

    def calculate_norm_stats(self, dataset):
        """Calculate mean and std for dataset normalization."""
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        mean_sum, std_sum, total_images = 0, 0, 0

        for data, _ in dataloader:
            mean_sum += data.mean(dim=[0, 2, 3])
            std_sum += data.std(dim=[0, 2, 3])
            total_images += 1

        self.means = (mean_sum / total_images).tolist()
        self.stds = (std_sum / total_images).tolist()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.means, std=self.stds),
        ])
        print(f"Calculated means: {self.means}, stds: {self.stds}")

    def train_epoch(self, dataloader):
        """Perform a training epoch."""
        train_losses = []
        correct = 0
        total = 0

        self.unet.train()
        for data, target in dataloader:
            data = self.transform(data).to(self.device)  # Apply transformation
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred = self.unet(data)
            loss = self.criterion(pred, target.float())
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            processed_preds = (pred.sigmoid() > self.threshold).int()
            correct += (processed_preds == target).sum().item()
            total += target.numel()

        avg_loss = np.mean(train_losses)
        accuracy = correct / total
        return avg_loss, accuracy

    def validation_epoch(self, dataloader):
        """Perform a validation epoch."""
        val_losses = []
        correct = 0
        total = 0

        self.unet.eval()
        with torch.no_grad():
            for data, target in dataloader:
                data = self.transform(data).to(self.device)  # Apply transformation
                target = target.to(self.device)

                pred = self.unet(data)
                loss = self.criterion(pred, target.float())
                val_losses.append(loss.item())

                processed_preds = (pred.sigmoid() > self.threshold).int()
                correct += (processed_preds == target).sum().item()
                total += target.numel()

        avg_loss = np.mean(val_losses)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, dataset, num_epochs=10, start_epoch=1):
        """Trains the model using the provided data and target. Saves the history of the training."""
    
        # Initialize the plot
        self.initialize_plot()

        # Split dataset
        total_size = len(dataset)
        val_size = int(0.15 * total_size)
        train_size = total_size - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_set.dataset.load_patches()
        val_set.dataset.load_patches()

        self.means = dataset.means
        self.stds = dataset.stds
        self.standardize = T.Normalize(mean=self.means, std=self.stds) 

        self.all_histories['means'] = self.means.tolist()
        self.all_histories['stds'] = self.stds.tolist()
    
        data_train = [x[0] for x in train_set]
        target_train = [x[1] for x in train_set]
        data_val = [x[0] for x in val_set]
        target_val = [x[1] for x in val_set]

        # Initial evaluation
        t, f, a = self.validation_epoch(data_train, target_train)
        self.add_history(t, f, a, "Train")
        t, f, a = self.validation_epoch(data_val, target_val)
        self.add_history(t, f, a, "Validation")
        self.print_history(0, num_epochs)
        self.update_plot(0)

        for epoch in range(start_epoch - 1, num_epochs):
            t, f, a = self.train_epoch(data_train, target_train)
            self.add_history(t, f, a, "Train")
            t, f, a = self.validation_epoch(data_val, target_val)
            self.add_history(t, f, a, "Validation")

            self.save_model(f"{self.model_name}_epoch_{epoch + 1}")
            self.print_history(epoch + 1, num_epochs)
            self.update_plot(epoch + 1)

        return self.all_histories

    
    def initialize_plot(self):
        """Initialize the plot for live training visualization."""
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 10))
        self.axs[0].set_title("Training & Validation Loss")
        self.axs[1].set_title("Training & Validation F1-Score")
        self.axs[2].set_title("Training & Validation Accuracy")
        self.axs[0].set_xlabel("Epochs")
        self.axs[1].set_xlabel("Epochs")
        self.axs[2].set_xlabel("Epochs")
        self.axs[0].set_ylabel("Loss")
        self.axs[1].set_ylabel("F1-Score")
        self.axs[2].set_ylabel("Accuracy")

    def update_plot(self, epoch):
        """Update the live plot with training and validation metrics."""
        clear_output(wait=True)

        # Epochs to display
        epochs = range(1, epoch + 2)

        # Plot Training & Validation Loss
        self.axs[0].plot(
            epochs,
            self.all_histories["Train"]["Train loss"][:epoch + 1],
            label="Train Loss",
            color="blue",
        )
        self.axs[0].plot(
            epochs,
            self.all_histories["Validation"]["Train loss"][:epoch + 1],
            label="Validation Loss",
            color="orange",
        )
        self.axs[0].legend()

        # Plot Training & Validation F1-Score
        self.axs[1].plot(
            epochs,
            self.all_histories["Train"]["F1_score"][:epoch + 1],
            label="Train F1-Score",
            color="blue",
        )
        self.axs[1].plot(
            epochs,
            self.all_histories["Validation"]["F1_score"][:epoch + 1],
            label="Validation F1-Score",
            color="orange",
        )
        self.axs[1].legend()

        # Plot Training & Validation Accuracy
        self.axs[2].plot(
            epochs,
            self.all_histories["Train"]["Accuracy"][:epoch + 1],
            label="Train Accuracy",
            color="blue",
        )
        self.axs[2].plot(
            epochs,
            self.all_histories["Validation"]["Accuracy"][:epoch + 1],
            label="Validation Accuracy",
            color="orange",
        )
        self.axs[2].legend()

        # Show the plot
        plt.tight_layout()
        plt.show()
