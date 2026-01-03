# UNI + MLP with Spatial Context

## Description
This pipeline uses the UNI foundation model as a frozen feature extractor
and trains a non-linear MLP head to predict MSI PCA components.

Spatial context is incorporated by aggregating features from neighboring
spots within the same tissue section.

## Key Components
- UNI (frozen)
- PCA on MSI (train-only)
- Context-aware feature aggregation
- MLP regression head

## Training
- Train/test split at slide level
- Weighted MSE loss
- AdamW optimizer

## Output
- Training curves
- PCA variance plots
- Scatter plots and spatial heatmaps
