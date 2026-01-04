# Pipeline 3: WSI to MSI Prediction using Graph Neural Networks (GNN)

## Overview
This pipeline implements an end-to-end framework for predicting spatial molecular profiles
(Mass Spectrometry Imaging – MSI) directly from Whole-Slide Histopathology Images (WSI).
The model leverages spatial relationships between tissue regions using a Graph Neural Network (GNN),
allowing molecular information to be inferred from histology alone during inference.

The pipeline is designed to operate in a weakly supervised setting where MSI data is available
only during training, while inference relies exclusively on WSI-derived features.

---

## Problem Setting
- **Input modality:** Whole-Slide Histopathology Images (WSI)
- **Target modality:** MSI molecular profiles (high-dimensional spectra)
- **Challenge:** Predicting high-dimensional molecular signals from visual tissue morphology
- **Constraint:** No MSI data is available at inference time

---

## Pipeline Architecture

### 1. Input Data
- Whole-Slide Histopathology Images (WSI)
- Spatially-resolved MSI measurements (available only during training)

---

### 2. Patch Extraction and Spatial Alignment
- Each MSI spot is aligned to a corresponding WSI region using spatial coordinates
- A fixed-size image patch (224×224) is extracted per MSI spot
- This enforces one-to-one correspondence between visual patches and molecular locations

---

### 3. Visual Feature Extraction
- A pretrained **ResNet-50** model is used as a frozen feature extractor
- Patch-level embeddings are extracted from the final convolutional layers
- No fine-tuning is performed to prevent overfitting and data leakage

---

### 4. MSI Preprocessing
- Raw MSI spectra are standardized
- Principal Component Analysis (PCA) is applied
- The model predicts MSI representations in PCA space instead of raw spectra
- PCA is fitted only on training data to preserve proper evaluation protocol

---

### 5. Spatial Graph Construction
- Each MSI spot / WSI patch is represented as a graph node
- Edges are constructed using k-nearest neighbors based on spatial coordinates
- The resulting graph captures local tissue spatial organization

---

### 6. Graph Neural Network (GNN)
- A **Graph Attention Network (GATv2)** is employed
- Multiple message-passing layers learn spatial dependencies between tissue regions
- Attention mechanisms allow the model to weight neighboring contributions adaptively
- Residual connections and normalization improve stability and convergence

---

### 7. Regression Head
- A Multi-Layer Perceptron (MLP) maps GNN embeddings to MSI PCA components
- The regression head produces predicted molecular profiles per spatial location

---

### 8. Training Strategy
- Training is performed at the slide level to prevent data leakage
- MSI PCA components serve as supervision targets
- A composite loss function is used:
  - Mean Squared Error (MSE)
  - Pearson correlation loss
  - Cosine similarity loss
- Model selection is based on validation performance

---

### 9. Inference
- During inference, only WSI patches and spatial coordinates are required
- The trained GNN predicts MSI PCA features for unseen slides
- No MSI measurements are used at test time

---

### 10. Evaluation and Visualization
The model is evaluated using multiple complementary metrics:
- **R² (coefficient of determination)**
- **RMSE (Root Mean Squared Error)**
- **F1-score** (after binarization in PCA space)
- **Pearson correlation**

Spatial heatmaps are generated to compare true and predicted molecular distributions,
providing qualitative assessment of spatial fidelity.

---

## Quantitative Results
Example evaluation results obtained on held-out slides:

- **R² Overall:** -2.3165  
- **RMSE Overall:** 2.5748  

These results reflect the inherent difficulty of inferring high-dimensional molecular
information solely from histopathological appearance and highlight the challenging
nature of cross-modal spatial prediction.

---

## Implementation Details
- Frameworks: PyTorch, PyTorch Geometric
- Backbone CNN: ResNet-50 (ImageNet pretrained, frozen)
- Graph Model: GATv2
- Dimensionality Reduction: PCA
- Training: GPU-supported, reproducible with fixed random seeds

---
<p align="center"><b>Figure 3. WSI-to-MSI Graph Neural Network (GNN) Pipeline Architecture and Evaluation Results</b></p>

<p align="center">
  <img width="1095" height="599" alt="Pipeline 3 Architecture" src="https://github.com/user-attachments/assets/ebb7fc3b-ca33-4bc3-8886-daebbb1b0acb">
</p>

