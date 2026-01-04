# Pipeline 1: ResNet50 vs UNI for MSI Prediction from Histopathology

## Overview
This pipeline predicts molecular profiles from Mass Spectrometry Imaging (MSI) using only H&E whole-slide histopathology images (WSI).
The main goal is to compare two pretrained visual encoders, ResNet50 and UNI, for mapping tissue morphology to molecular information.

The experiment is designed as a single-slide, spot-level regression task, where MSI data is used only during training and predicted during testing.

---

## Method Description

### Input Data
- Whole-slide histopathology images (H&E stained)
- Corresponding MSI molecular spectra (available only during training)

Each MSI spatial spot is aligned with a WSI patch extracted from the tissue image.

---

### Preprocessing
- WSI patches are resized to 224 × 224 pixels
- MSI spectra are standardized using StandardScaler
- Dimensionality reduction is applied using Principal Component Analysis (PCA)

Number of PCA components: 100

---

### Feature Extraction
Two pretrained encoders are used to extract image features:

- ResNet50:
  - General-purpose image encoder
  - Pretrained on ImageNet

- UNI:
  - Histopathology-specific foundation model
  - Pretrained on large-scale pathology datasets

Each encoder outputs a fixed-length feature vector per WSI patch.

---

### Regression Model
A Ridge Regression model (RidgeCV) is trained to map image features to MSI PCA components.
Regularization strength is selected automatically using cross-validation.

---

### Training and Testing Strategy
- Train/test split is performed within the slide
- During training:
  - WSI image features and MSI PCA features are used
- During testing:
  - Only WSI image features are provided
  - MSI molecular profiles are predicted

---

## Results

Performance is evaluated using Root Mean Squared Error (RMSE) and R-squared (R²).

- ResNet50:
  - RMSE = 1.577
  - R² = -0.724

- UNI:
  - RMSE = 1.325
  - R² = -0.173

---

## Interpretation
UNI outperformed ResNet50 by achieving lower RMSE and higher R² values.
This suggests that histopathology-specific representations are more suitable for predicting molecular profiles than general image features.

However, negative R² values indicate that the task is challenging due to limited training data and high biological variability.

---
<img width="1071" height="636" alt="image" src="https://github.com/user-attachments/assets/661efd1b-44dd-44a9-8760-4cfbe163a7a3" />
<p align="center">
<b>Figure 1.</b> Computational pathology pipeline for predicting MSI molecular profiles from H&E whole-slide images.
During training, paired WSI patches and MSI PCA features are used to train a regression model.
During inference, only WSI patches are provided and MSI molecular profiles are predicted without MSI measurements.
</p>
