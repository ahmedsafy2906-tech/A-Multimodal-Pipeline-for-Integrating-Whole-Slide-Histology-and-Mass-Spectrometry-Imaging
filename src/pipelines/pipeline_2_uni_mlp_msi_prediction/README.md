<img width="1088" height="640" alt="image" src="https://github.com/user-attachments/assets/c55bde6e-99a8-4f72-bcf8-b492afe93ca0" />## Pipeline 2 — UNI + Context-Aware MLP for MSI Prediction

This pipeline implements an end-to-end **multimodal computational pathology framework** for predicting **Mass Spectrometry Imaging (MSI) molecular profiles** directly from **whole-slide histopathology images (WSI)**. The model leverages a frozen **UNI histopathology foundation model** for feature extraction and a **non-linear, context-aware MLP regression head** for molecular prediction.

---

### Pipeline Overview

The pipeline is designed with a strict **training–inference separation** to prevent data leakage:

- **Training:** Both WSI patches and MSI data are available  
- **Inference:** Only WSI patches are used; MSI is fully predicted

---

### Methodology

**1. Input Data**  
- Whole-slide H&E histopathology images (WSI)  
- Spot-level MSI spectra (available only during training)

Each MSI spot corresponds to a spatial location on the WSI.

**2. Preprocessing**  
- WSIs are divided into fixed-size patches of **224 × 224 pixels** centered at MSI spot coordinates  
- MSI spectra are normalized on a slide level  
- Dimensionality reduction is performed using **Principal Component Analysis (PCA)**  
- PCA is fitted **exclusively on training slides** to ensure no data leakage

**3. Feature Extraction (Frozen Encoder)**  
- Encoder: **UNI (Vision Transformer, MahmoodLab)**  
- Pretrained on large-scale histopathology data  
- All encoder weights are frozen  
- Each image patch is mapped to a high-dimensional feature vector

**4. Spatial Context Modeling**  
For each MSI spot:
- The feature vector of the central patch is extracted  
- The `k = 6` nearest neighboring spots are identified spatially  
- Neighbor features are averaged  
- Central and neighbor features are concatenated to form a context-aware representation  

This allows the model to capture **local tissue architecture and spatial organization**.

**5. Non-Linear Regression Head**  
- A Multi-Layer Perceptron (MLP) is used to predict MSI PCA components  
- Architecture:  
  `Input → 1024 → 512 → Output (PCA-MSI)`  
- Includes Batch Normalization, ReLU activations, and Dropout  
- Optimization is performed using a **PCA-variance–weighted MSE loss**, prioritizing high-variance molecular components

**6. Training Strategy**  
- Training is performed across multiple slides  
- Slide-level train/test split is enforced  
- Best model is selected based on validation performance

**7. Inference**  
- Only WSI patches are provided
- UNI features and spatial context are computed  
- The MLP predicts MSI representations in PCA space  
- Final output corresponds to predicted molecular profiles without MSI measurements

---

### Quantitative Results

- **RMSE:** 2.0590  
- **R²:** −0.0046  

These results reflect the inherent difficulty of predicting high-dimensional molecular information from histology alone and emphasize the challenging nature of cross-modal inference in computational pathology.
<p align="center">
  <img width="1088" height="640" src="https://github.com/user-attachments/assets/16afb2ff-b4a5-42dd-805c-4d382664853f" />
</p>

<p align="center">
  <b>Figure 2.</b> End-to-end pipeline for predicting MSI molecular profiles from 
  whole-slide histopathology images using a frozen UNI foundation model, 
  spatial context aggregation, and a non-linear MLP regression head.
</p>





