# ResNet50 vs UNI – MSI Prediction Baseline

## Description
This pipeline compares two pretrained vision encoders:
- ResNet50 (ImageNet pretrained)
- UNI (histopathology foundation model)

Both models are used to extract features from WSI patches,
which are then mapped to MSI signals using PCA + Ridge Regression.

## Steps
1. Patch extraction from WSI
2. Spatial alignment with MSI spots
3. Feature extraction (ResNet50 / UNI)
4. PCA compression of MSI (train only)
5. Ridge regression
6. RMSE and R² evaluation

## Output
- RMSE and R² per slide
- PCA variance plots
- Prediction vs ground truth plots

## Notes
This pipeline serves as a baseline comparison before introducing
non-linear and spatial models.
