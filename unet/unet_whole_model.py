import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from dataset import *
from utils import *
import json
from unet_model import UNet
from torcheval.metrics import BinaryF1Score as F1_score
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os
import GPUtil

class Model:
    def __init__(self, model_name, lr) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = UNet(n_channels=4, n_classes=1)
        self.unet.to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.threshold = 0.
        self.model_name = model_name
        self.f1 = F1_score(threshold=self.threshold, device = self.device)
        self.batch_size = 13
        self.all_histories = {
            "Train": {"Train loss": [], "F1_score": [], "Accuracy": []},
            "Validation": {"Train loss": [], "F1_score": [], "Accuracy" : []},
        }
        self.model_path = f"/{self.model_name}/"

        self.means = None
        self.stds = None
        self.standardize = None
        self.batch_norm = nn.BatchNorm2d(num_features=4)

    def get_history(self):
        """Returns the training (and validation history if validation was used) of the current model.

        Returns:
            Dictionary: Dictionary containing all the losses and F1-scores of the training.
        """
        return self.all_histories

    def get_best_epoch(self):
        """Returns the epoch with the best F1-score.

        Returns:
            int: Epoch with the best F1-score.
        """
        #selecting best epoch on maximizing sum on validation f1
        return np.argmax(np.array(self.all_histories["Validation"]["F1_score"]))
    
    def get_best_f1_accuracy(self):
        """Returns the F1-score and accuracy of the best epoch.

        Returns:
            _type_: F1-score and accuracy of the best epoch.
        """
        epoch = self.get_best_epoch()
        return self.all_histories["Validation"]["F1_score"][epoch], self.all_histories["Validation"]["Accuracy"][epoch]

    def training_step(self, data, target):

        self.optimizer.zero_grad()
        #prediction
        target_pred = self.unet(data)
        #calculate the loss
        target = target.to(torch.float32)
        loss = self.criterion(target_pred, target)
        #backpropagation
        loss.backward()
        #model update
        self.optimizer.step()

        return loss, target_pred


    def train_epoch(self, data_list, target_list):
        """Performs a training epoch using stochastic gradient descent and back propagation over the provided data loader.

        Args:
            data_loader (_type_): Data to be used for the training epoch

        Returns:
            _type_: Train loss and F1-score of the epoch
        """
        train_losses = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        correct = 0
        incorrect = 0 
        total_samples = 0

        self.unet.train()


        for batch_idx, (data, target) in enumerate(zip(data_list, target_list)):
                batch_size = data.size(0)  # Get batch size
                total_samples += batch_size

                #move data to GPU
                data = data.to(self.device)
                target = target.to(self.device)

                #batch normalization 
                data = data.float()
                data = self.batch_norm(data)

                #predict + calculate loss
                loss, train_pred = self.training_step(data, target)
                train_losses.append(loss.cpu().detach().numpy())

                processed_preds =torch.where(train_pred < self.threshold, 0 ,1)
                correct += (processed_preds == target).sum().detach().cpu().item()
                incorrect += (processed_preds != target).sum().detach().cpu().item()
                true_positives += ((processed_preds == target) & (target == 1)).sum().detach().cpu().item()
                false_positives += ((processed_preds != target) & (target == 0)).sum().detach().cpu().item()
                false_negatives += ((processed_preds != target) & (target == 1)).sum().detach().cpu().item()



        
        train_loss= float(np.stack(train_losses).mean())
        
        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        
        accuracy = correct / (correct + incorrect)

        
        torch.cuda.empty_cache()

        return train_loss, f1, accuracy
    
    def evaluation_step(self, data, target):
        # Passer les données d'entrée à travers le modèle
        raw_pred = self.unet(data)
        # Convertir la cible en type de données float32 si nécessaire
        target = target.to(torch.float32)
        # Calculer la perte entre les prédictions brutes et la cible
        loss = self.criterion(raw_pred, target)

        return loss, raw_pred


    def validation_epoch(self, data_list, target_list):
        """Performs a validation epoch on the provided data loader. Predicts the values using the current state of the model and computers the validation loss and validation F1-score.

        Args:
            data_loader (_type_): Data to use for validation epoch.

        Returns:
            _type_: Validation loss and validation F1-score
        """
        val_losses = []
        correct = 0
        incorrect = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        self.unet.eval()
        with torch.no_grad():
           for data, target in zip(data_list, target_list):
                data = data.float()
        
                data = data.to(self.device)
                target = target.to(self.device)
                loss, val_pred = self.evaluation_step(data, target)

                val_losses.append(loss.cpu().detach().numpy())

                processed_preds =torch.where(val_pred < self.threshold, 0 ,1)
                correct += (processed_preds == target).sum().detach().cpu().item()
                incorrect += (processed_preds != target).sum().detach().cpu().item()
                true_positives += ((processed_preds == target) & (target == 1)).sum().detach().cpu().item()
                false_positives += ((processed_preds != target) & (target == 0)).sum().detach().cpu().item()
                false_negatives += ((processed_preds != target) & (target == 1)).sum().detach().cpu().item()



        val_loss = float(np.stack(val_losses).mean())
    
        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
        accuracy = correct / (correct + incorrect)
        
        
        torch.cuda.empty_cache()
        return val_loss, f1, accuracy

    def add_history(self, t, f, a, type):
        """Adds an input to the history of the model.

        Args:
            t (_type_): loss of the epoch
            f (_type_): f1-score of the epoch
            a (_type_)): accuracy of epoch
            type (_type_): Train or validation epoch (default: "Train")
        """
        self.all_histories[type]["Train loss"].append(t)
        self.all_histories[type]["F1_score"].append(f)
        self.all_histories[type]["Accuracy"].append(a)

    def print_history(self,epoch, num_epochs):
      print(
                "Epoch: {:d}/{:d} Train_loss: {:.5f}, Train_F1: {:.5f}, Train_Accuracy: {:.15f}, Val_loss: {:.5f}, Val_F1: {:.5f}, Val_Accuracy: {:.5f}".format(
                    epoch,
                    num_epochs,
                    self.all_histories["Train"]["Train loss"][epoch],
                    self.all_histories["Train"]["F1_score"][epoch],
                    self.all_histories["Train"]["Accuracy"][epoch],
                    self.all_histories["Validation"]["Train loss"][epoch],
                    self.all_histories["Validation"]["F1_score"][epoch],
                    self.all_histories["Validation"]["Accuracy"][epoch],
                )
            )


    def train(self, dataset ,num_epochs=10, start_epoch = 1):
        """Trains the model using the provided data and target. Saves the history of the training.
        """

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
        
        data_train= []
        target_train = []
        for x in train_set:
            data_train.append(x[0])
            target_train.append(x[1])
        print(data_train[0])
        data_val = []
        target_val = []
        for x in val_set:
            data_val.append(x[0])
            target_val.append(x[1])

        print("From initial model:")
        t, f, a = self.validation_epoch(data_train, target_train)
        self.add_history(t, f, a, "Train")
        t, f, a = self.validation_epoch(data_val, target_val)
        self.add_history(t, f, a, "Validation")
        self.print_history(0,num_epochs)

        for epoch in range(start_epoch-1,num_epochs):
            t, f, a = self.train_epoch(data_train, target_train)
            self.add_history(t, f, a, "Train")
            t, f, a = self.validation_epoch(data_val, target_val)
            self.add_history(t, f, a, "Validation")
            self.save_model(f"{self.model_name}_epoch_{epoch+1}")
            self.print_history(epoch + 1,num_epochs)

        return self.all_histories
    
    def predict(self, img):
        """Predicts the segmentation of the input image using the current state of the model.

        Args:
            img (_type_): Input image to be segmented in tensor format

        Returns:
            _type_: Segmentation of the input image in tensor format
        """
        img_input = img.unsqueeze(0)
        self.unet.eval()
        with torch.no_grad():
            pred = self.unet(self.standardize(img_input).to(self.device))
        
        pred = pred.squeeze(1)
        pred = torch.where(pred < self.threshold, 0 ,1).cpu()

        return pred
    
    def plot_prediction(self, img_lab_dict):
        """Plots the input image and it's segmentation.

        Args:
            img (_type_): Input image to be segmented in tensor format
        """
        img = img_lab_dict['img']
        standardized_img = self.standardize(img_lab_dict['img'])
        
        gt = img_lab_dict['gt']

        pred = self.predict(standardized_img) 

        standardized_rgb_img = (standardized_img.numpy()[:3] * 255).astype(np.uint8).transpose(1, 2, 0)
        rgb_img = (img.numpy()[:3] * 255).astype(np.uint8).transpose(1, 2, 0)
        rgb_pred = pred.numpy().astype(np.uint8).squeeze(0) * 255
        rgb_gt = gt.numpy().astype(np.uint8).squeeze(0) * 255
        
        accuracy = (pred == gt).sum().item() / (pred.numel())
        false_negatives = ((pred != gt) & (gt == 1)).sum().item() 
        false_positives = ((pred != gt) & (gt == 0)).sum().item() 
        true_positives = ((pred == gt) & (gt == 1)).sum().item()

        f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

        _ , axs = plt.subplots(1, 4, figsize=(20, 10))
        axs[0].imshow(rgb_img)
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(standardized_rgb_img)
        axs[1].set_title("Standardized image")
        axs[1].axis("off")
        axs[2].imshow(rgb_pred, cmap='gray')
        axs[2].set_title("Prediction")
        axs[2].axis("off")
        axs[3].imshow(rgb_gt, cmap='gray')
        axs[3].set_title("Groundtruth")
        axs[3].axis("off")
        plt.show()

        print(f"On this image: Accuracy: {accuracy:.5f}, F1-score: {f1:.5f}")
    
    
    def plot_history(self):

        x_axis = range(1, len(self.all_histories['Train']['Train loss']))
        _, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

        color_loss = '#87BCDE'  # Blue color
        color_other = '#805E73'  # Purple color

        axs[0].plot(x_axis, self.all_histories['Train']['Train loss'][1:], label='Train', color=color_loss)
        axs[0].plot(x_axis, self.all_histories['Validation']['Train loss'][1:], label='Validation', color=color_other)
        axs[0].set_title("Losses")
        axs[0].legend()

        axs[1].plot(x_axis, self.all_histories['Train']['F1_score'][1:], label='Train', color=color_loss)
        axs[1].plot(x_axis, self.all_histories['Validation']['F1_score'][1:], label='Validation', color=color_other)
        axs[1].set_title("F1 scores")
        axs[1].legend()

        axs[2].plot(x_axis, self.all_histories['Train']['Accuracy'][1:], label='Train', color=color_loss)
        axs[2].plot(x_axis, self.all_histories['Validation']['Accuracy'][1:], label='Validation', color=color_other)
        axs[2].set_title("Accuracies")
        axs[2].legend()

        # Setting x-axis ticks and label
        plt.xticks(x_axis, ['Epoch {}'.format(i) for i in x_axis])
        plt.xlabel('Epoch')

    def test_dataset(self, test_dataset):
        """Tests the current model on the provided dataset."""
        test_dataset.load_images_and_gts()
        
        loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True
        )
        _, f, a = self.validation_epoch(loader)

        test_dataset.deload()

        return f,a

    def save_model(self, model_name):
        """Saves the current state of the model. Automatically puts it in the correct folder using the proper file extensions.

        Args:
            model_name (_type_): Name of the model to use for save file.
        """


        path = self.model_path + "{:s}.{:s}"
        os.makedirs(self.model_path, exist_ok=True)
        
        torch.save(self.unet.state_dict(), path.format(model_name, "pth"))
    
    
        # Convert tensor to list before serializing
        all_histories_serializable = {
            "train_losses": self.all_histories["train_losses"].tolist(),
            "train_accuracies": self.all_histories["train_accuracies"].tolist(),
            "validation_losses": self.all_histories["validation_losses"].tolist(),
            "validation_accuracies": self.all_histories["validation_accuracies"].tolist()
        }
        
        json_dict = json.dumps(all_histories_serializable)
        file = open(path.format(model_name, "json"), "w")
        file.write(json_dict)
        file.close

    def load_model(self, model_name):
        """Loads a saved model and it's history into the current model. Overwrites any prior training and history of the current model.

        Args:
            model_name (_type_): Name of the model to load (automatic path finding as for save_model()).
        """
        path = self.model_path + "{:s}.{:s}"
        self.unet.load_state_dict(
            torch.load(
                path.format(model_name, "pth"),
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )
        self.unet.eval()
        with open(path.format(model_name, "json")) as file:
            data = file.read()
            self.all_histories = json.loads(data)
        self.means = torch.tensor(self.all_histories['means'])
        self.stds = torch.tensor(self.all_histories['stds'])
        self.standardize = T.Normalize(mean=self.means, std=self.stds) 
