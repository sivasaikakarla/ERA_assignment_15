# U-Net Segmentation with Creative Loss

This repository contains the implementation of a U-Net architecture specifically designed for semantic segmentation of the Oxford-IIIT Pet Dataset.

## Project Overview
The assignment required training a U-Net from scratch using 4 different configurations to analyze the impact of architectural choices (Max Pooling vs. Strided Conv) and loss functions.

## Creative Loss Details: Dice Loss
For this project, I implemented **Dice Loss** as a creative alternative to the standard Binary Cross Entropy (BCE).

### Why Dice Loss?
Standard loss functions like BCE treat all pixels equally. In pet segmentation, the "Background" class often dominates the image (e.g., 80% background, 20% pet). A model can cheat by predicting "background" everywhere. 

**Dice Loss** solves this by optimizing the **Dice Coefficient (F1 Score)** directly. It focuses on the overlap between the predicted mask and the ground truth mask, making it robust to class imbalance.

### Mathematical Formulation
The Dice Loss is calculated as:

$$Loss_{Dice} = 1 - \frac{2 \sum_{i} y_{true} \cdot y_{pred}}{\sum_{i} y_{true} + \sum_{i} y_{pred} + \epsilon}$$

Where $\epsilon$ is a smoothing factor to prevent division by zero.

## Results
- **App Demo:** [https://huggingface.co/spaces/siva200/Unet]
- **Best Model:** The model utilizing Strided Convolutions + Upsampling + Dice Loss achieved the sharpest segmentation boundaries.
