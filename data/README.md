## Dataset

This project uses data from the following publicly available dataset:

**Spatial Multimodal Analysis of Transcriptomes and Metabolomes in Tissues**  
Mendeley Data, 2023  
DOI: https://doi.org/10.17632/w7nw4km7xd.1

### Modalities
The dataset provides paired and spatially resolved measurements from:
- Whole-Slide Histology images (H&E)
- Mass Spectrometry Imaging (MSI, MALDI-MSI)
- Spatial transcriptomics (10x Visium)

The data include samples from **mouse and human brain tissue**, with molecular measurements covering **small metabolites, neurotransmitters, and lipids**.

### SMA-aligned Data (Used in This Project)

In this project, we exclusively use the **`sma` folder** from the dataset.

The `sma` subset corresponds to data generated using the **Spatial Multimodal Analysis (SMA) protocol**, where:
- All modalities (H&E, MSI, and transcriptomics) are acquired from the **same tissue section**
- All data are **spatially registered** in a shared coordinate system
- Each MSI spot is directly aligned with its corresponding location in the WSI

This strict spatial alignment enables:
- Patch-level extraction from whole-slide histology
- Spot-level molecular supervision from MSI
- Reliable cross-modal learning without additional registration steps

Using the SMA-aligned data ensures a **one-to-one correspondence between tissue morphology and molecular signals**, which is essential for training and evaluating WSI-to-MSI prediction models.

### Experimental Protocol
The SMA workflow combines:
- MALDI-MSI for metabolite imaging
- H&E staining for tissue morphology
- Visium spatial transcriptomics for gene expression

All measurements are performed on the **same tissue section**, preserving spatial correspondence across modalities.
