# Spatial GNN for WSI-to-MSI Prediction

## Description
This pipeline models spatial relationships between tissue spots
using a graph neural network (GNN).

Each spot is represented as a node, with edges defined by k-nearest
neighbors in tissue space.

## Model
- ResNet50 feature extraction
- PCA-compressed MSI targets
- GATv2-based GNN
- Composite loss (MSE + Pearson + Cosine)

## Purpose
This model extends patch-level prediction by explicitly modeling
spatial tissue structure.

## Status
Experimental / Research extension
