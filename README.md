# A Multimodal Pipeline for Integrating Whole-Slide Histology and Mass Spectrometry Imaging

## Overview
This project explores multiple deep learning pipelines for predicting spatial molecular profiles from whole-slide histopathology images (WSI).  
Mass Spectrometry Imaging (MSI) data is used only during training as supervision, while inference relies solely on histology images.

The project compares different modeling strategies, including CNN-based feature extraction, MLP regression, and graph neural networks for spatial modeling.

## Project Structure
- `src/`: Source code for all pipelines
- `docs/`: Project documentation and design details
- `data/`: Instructions for obtaining datasets
- `requirements.txt`: Python dependencies

## Installation
```bash
pip install -r requirements.txt
