
## Overview
This project investigates whether spatial molecular information obtained from Mass Spectrometry Imaging (MSI) can be inferred directly from routine whole-slide histopathology images (WSI). While MSI provides rich molecular insight, it is costly and rarely available in clinical settings. In this work, MSI is used only during training as supervision, while inference relies solely on histology images.

The project explores multiple deep learning pipelines with increasing levels of spatial awareness to study how tissue morphology relates to molecular composition.

---

## Core Idea
- Learn a mapping from histological morphology to molecular signals.
- Use paired WSI–MSI data during training only.
- Predict spatial molecular patterns from histology alone at inference time.
- Evaluate the effect of spatial modeling on prediction quality.
<p align="center">
  <strong>Figure. Core idea of learning a unified mapping from histological morphology (WSI) to spatial molecular patterns (MSI).  
  MSI data is used only during training, while inference relies solely on histology images.</strong>
</p>

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/068bab9d-42b9-4bec-b4c0-ec45eb15f5a0"
    width="900"
  />
</p>

---

## Pipeline Overview

### Pipeline 1 — CNN Feature Extraction + Regression
This baseline pipeline extracts patch-level visual features from whole-slide images using convolutional neural networks. Each patch is treated independently, and MSI-derived targets are predicted using a regression model. No explicit spatial relationships between patches are modeled, making this pipeline a reference point for comparison.

---

### Pipeline 2 — Frozen Foundation Model + MLP Regression
In this pipeline, a pretrained histopathology foundation model (such as UNI) is used as a frozen feature extractor. The extracted high-level visual embeddings are passed to a multi-layer perceptron (MLP) to predict PCA-compressed MSI targets. This approach leverages strong pretrained representations but still lacks explicit spatial reasoning.

---

### Pipeline 3 — WSI-to-MSI Prediction using Graph Neural Networks
This pipeline explicitly models tissue structure as a spatial graph. Each node represents a spatial spot or patch aligned between WSI and MSI, while edges encode spatial neighborhood relationships. A Graph Attention Network (GATv2) propagates information between neighboring tissue regions, enabling the model to capture local tissue architecture and spatial dependencies that are not accessible to independent patch-based models.

---

## Data Representation

### Histopathology (WSI)
Whole-slide images are divided into fixed-size patches aligned to MSI spatial coordinates. Visual features are extracted using pretrained convolutional or foundation models, producing compact representations of tissue morphology at each spatial location.

### Mass Spectrometry Imaging (MSI)
MSI provides high-dimensional molecular measurements for each spatial spot. These measurements are normalized and compressed using Principal Component Analysis (PCA). The resulting PCA components serve as regression targets during model training.
<div align="center">

<img
  src="https://github.com/user-attachments/assets/e28ac224-ce63-416e-95e3-98f9bef9f1da"
  width="1120"
  height="633"
  alt="WSI and MSI Complementary Technologies Architecture"
/>

<p><b>Figure.</b> WSI & MSI as complementary technologies for tissue analysis.  
Routine histopathology (WSI) captures tissue morphology, while Mass Spectrometry Imaging (MSI) provides spatially resolved molecular profiles.  
The proposed framework bridges morphology and molecular composition to enable MSI prediction from WSI.</p>

</div>
  
---
<div align="center">
  <img width="1206" height="598"
       src="https://github.com/user-attachments/assets/a3cb48c3-1571-4c23-98c1-a7f0d250ac31"
       alt="WSI–MSI spot-level alignment illustration" />

  <p><b>Figure:</b> Spot-level alignment between whole-slide histology (WSI) and mass spectrometry imaging (MSI).  
  Each MSI spatial spot is mapped to a corresponding WSI patch using pixel coordinates, forming the fundamental linking unit between tissue morphology and molecular measurements.</p>
</div>

## Training Strategy
- Slide-level train/test splitting is applied to prevent data leakage.
- MSI data is used strictly during training and never during inference.
- PCA and normalization parameters are fitted using training data only.
- Models are optimized using regression-based loss functions.
- For graph-based models, spatial graphs are constructed using k-nearest-neighbor relationships between tissue spots.

---

## Evaluation Metrics
Model performance is evaluated using multiple complementary metrics:
- **R² (Coefficient of Determination)** to assess explained variance.
- **RMSE (Root Mean Squared Error)** to quantify prediction error..

---

## Results Summary
Overall predictive performance is modest, which is expected given the intrinsic difficulty of inferring high-dimensional molecular signals from histology alone. Low or negative R² values reflect strong biological heterogeneity, limited training data, and noise in MSI measurements. Graph-based models show improved spatial coherence compared to patch-independent approaches, highlighting the importance of spatial context.

---

## Limitations
- Very limited number of paired WSI–MSI slides.
- High inter-slide biological variability.
- Noisy and sparse MSI supervision signals.
- PCA compression may discard biologically meaningful variation.
- Patch-based modeling may miss global tissue context.

---

## Future Directions
- Expand datasets across patients, tissues, and cohorts.
- Integrate additional spatial omics modalities.
- Predict region-level molecular patterns instead of single spots.
- Fine-tune foundation models on paired multimodal data.
- Explore hierarchical or multi-scale graph architectures.
- Replace PCA with biologically informed latent representations.

---

## Key Takeaway
This project demonstrates a structured exploration of multimodal learning for linking tissue morphology to molecular information. While quantitative performance is limited, the pipelines establish a strong experimental framework and emphasize the critical role of spatial modeling when learning cross-modal relationships.

---

## Disclaimer
This project is intended for research and educational purposes only and is not suitable for clinical use.
