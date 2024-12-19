# Ant Morphology Segmentation with U-Net

## Overview
This project aims to automate the segmentation of ant body parts (head, thorax, and abdomen) from RGB images using a U-Net-based convolutional neural network (CNN). The solution is designed for researchers and biologists to facilitate morphological studies, taxonomy, and ecological research by reducing manual annotation efforts.

Key features of the project:
- Uses a U-Net architecture for pixel-wise segmentation.
- Includes preprocessing steps for dataset preparation and augmentation.
- Implements training, validation, and evaluation workflows with accuracy and F1 score metrics.
- Provides scripts for data loading, model definition, and result visualization.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [U-Net Model](#u-net-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)

## Installation 

Clone this repository, in order to have run the different code

```bash
git clone https://github.com/heinkele/CS-433-Project_2_Ant_Segmentation
```

## Project Structure and Usage
- **`ants_dataset.py`**:
  Contains the `AntsDataset` class for loading and preprocessing ant images and masks. Supports augmentation techniques like rotation, flipping, and brightness adjustment.
  
- **`unet.py`**:
  Defines the U-Net architecture with encoder, decoder, bottleneck, and skip connections. This model is tailored for 4-class segmentation tasks.
  
- **`utils_final.py`**:
  Utility functions for evaluation, saving models, computing metrics (accuracy, F1 score), and visualizing predictions.

- **`main_final.ipynb`**:
  A Jupyter Notebook implementing the training and evaluation pipeline. This notebook integrates preprocessing, model training, hyperparameter tuning, and result visualization.

- **`ML_Project2-4.pdf`**:
  The project report detailing the methodology, U-Net architecture, results, and ethical assessment.

  Run the script main.ipynb to preprocess the data, implement/train the models and get the final predictions. Pay attention to adapt the different pathway in the part : "Creation of the directories and data splitting", "Model Training   function save_model()" and "Save the trained model for the best epoch",  depending on where the data are store or when you want to store the result.
---

## Dataset
The dataset consists of 190 manually annotated grayscale masks paired with the associated RGB images of ants coming from Antweb. Each mask classifies pixels into 4 classes : Background, Head, Thorax and Abdomen


### Preprocessing Steps
1. Resizing all images and masks to `256x256`.
2. Normalizing images to have zero mean and unit variance.
3. Data augmentation techniques:
   - Random rotation (`90°`, `180°`, `270°`)
   - Horizontal and vertical flipping
   - Gaussian Blur (intensity sigma between 0.1 and 0.5, on 50% of the images)

## U-Net Model

The U-Net model is the core of this project and is specifically designed for segmentation tasks. The architecture consists of the following key components:

1. **Encoder**:
   - The encoder extracts hierarchical features using a series of convolutional and max-pooling layers.
   - Each layer reduces the spatial dimensions while increasing the depth of feature maps.

2. **Bottleneck**:
   - The bottleneck layer acts as a bridge between the encoder and decoder, capturing high-level abstract features.
   - This layer contains the highest number of filters, enabling the model to learn complex patterns.

3. **Decoder**:
   - The decoder reconstructs the spatial resolution of the input image using transposed convolution layers.
   - Skip connections from the encoder are used to retain spatial details lost during down-sampling.

4. **Skip Connections**:
   - These connections transfer high-resolution feature maps from the encoder to the decoder, ensuring precise localization.
   - They help maintain fine-grained details, which are critical for distinguishing small or overlapping regions like ant body parts.

5. **Output Layer**:
   - A 1x1 convolutional layer outputs class probabilities for each pixel. The softmax activation function ensures these probabilities sum to one for all classes.


---

## Evaluation Metrics

To assess the performance of the U-Net model, the following metrics were used:

1. **Pixel Accuracy**:
   - Measures the proportion of correctly classified pixels in the segmentation mask.
   - Formula:
     \[
     A = \frac{TP + TN}{TP + TN + FP + FN}
     \]
   - Where:
     - \( TP \): True Positives
     - \( TN \): True Negatives
     - \( FP \): False Positives
     - \( FN \): False Negatives

2. **F1 Score**:
   - The F1 score is the harmonic mean of precision and recall, particularly useful for imbalanced classes.
   - Formula:
     \[
     F1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
     \]
   - The F1 Score helps in evaluating the model's ability to segment smaller regions like the head and thorax.

These metrics are computed during both validation and testing phases to ensure the model performs well on unseen data. Detailed implementation can be found in `utils_final.py`.


---

## Conclusion

This project successfully demonstrates the application of the U-Net architecture for segmenting ant body parts from RGB images. By leveraging robust preprocessing, data augmentation, and carefully chosen evaluation metrics, the model achieves high accuracy and generalization. 



### Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- NumPy
- Matplotlib
- tqdm
- scikit-learn


### Contributors 
- Philippine Laroche
- Nils Manni
- Hugo Heinkele
