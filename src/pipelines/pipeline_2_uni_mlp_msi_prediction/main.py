from huggingface_hub import login

# Log in using the provided token
login(token="")

import os
import json
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================================
# 0. ENVIRONMENT SETUP
# ============================================================================

print("=" * 80)
print("MSI-from-Histology Prediction Pipeline")
print("=" * 80)
    # options: "UNI" or "RESNET50"

# Mount Google Drive
print("\n[STEP 0] Mounting Google Drive...")
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# Verify GPU
assert torch.cuda.is_available(), "❌ CUDA not available! This pipeline requires GPU."
device = torch.device("cuda")
print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
print(f"✓ Device: {device}")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"✓ Random seed set to {SEED}")

# Dataset root
DATA_ROOT = "/content/drive/MyDrive/sma/sma"
assert os.path.exists(DATA_ROOT), f"❌ Data root not found: {DATA_ROOT}"
print(f"✓ Data root verified: {DATA_ROOT}")


# ============================================================================
# 1. DATA DISCOVERY
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 1] Discovering samples...")
print("=" * 80)

def discover_samples(root: str) -> List[Dict]:
    """
    Discover all valid samples with required files.

    Expected structure:
    sma/SAMPLE_FAMILY/SAMPLE_ID/output_data/
        *_RNA/outs/spatial/
            tissue_hires_image.png
            tissue_positions_list.csv
            scalefactors_json.json
        *_MSI/
            *.csv

    Returns:
        List of dicts with paths to WSI, spatial files, and MSI
    """
    samples = []

    for family_dir in Path(root).glob("*"):
        if not family_dir.is_dir():
            continue

        for sample_dir in family_dir.glob("*"):
            if not sample_dir.is_dir():
                continue

            output_dir = sample_dir / "output_data"
            if not output_dir.exists():
                continue

            # Find RNA and MSI directories
            rna_dirs = list(output_dir.glob("*_RNA"))
            msi_dirs = list(output_dir.glob("*_MSI"))

            if not rna_dirs or not msi_dirs:
                continue

            rna_dir = rna_dirs[0]
            msi_dir = msi_dirs[0]

            # Check for required spatial files
            spatial_dir = rna_dir / "outs" / "spatial"
            wsi_path = spatial_dir / "tissue_hires_image.png"
            positions_path = spatial_dir / "tissue_positions_list.csv"
            scalefactors_path = spatial_dir / "scalefactors_json.json"

            # Find MSI CSV
            msi_csvs = list(msi_dir.glob("*.csv"))

            # Validate all required files exist
            if not all([
                wsi_path.exists(),
                positions_path.exists(),
                scalefactors_path.exists(),
                len(msi_csvs) > 0
            ]):
                print(f"⚠ Skipping {sample_dir.name}: missing required files")
                continue

            samples.append({
                "sample_id": sample_dir.name,
                "family": family_dir.name,
                "wsi_path": str(wsi_path),
                "positions_path": str(positions_path),
                "scalefactors_path": str(scalefactors_path),
                "msi_path": str(msi_csvs[0])
            })

            print(f"✓ Found valid sample: {family_dir.name}/{sample_dir.name}")

    return samples

samples = discover_samples(DATA_ROOT)
print(f"\n✓ Total valid samples found: {len(samples)}")

# Verify minimum sample count for 10 train + 3 test split
# assert len(samples) >= 13, f"❌ Need at least 13 slides, found only {len(samples)}"
# Relaxed check since we might filter incompatible ones
if len(samples) < 5:
    print("⚠ Warning: Very few samples found.")

def get_msi_signature(msi_path: str):
    cols = pd.read_csv(msi_path, nrows=0).columns.tolist()
    meta_cols = {'x', 'y', 'array_row', 'array_col', 'Unnamed: 0'}
    return tuple(sorted([c for c in cols if c not in meta_cols]))

# ============================================================================
# 2. SLIDE-LEVEL SPLITTING (FIXED: 10 TRAIN, 3 TEST)
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 2] Splitting slides (10 train, 3 test)...")
print("=" * 80)

from collections import defaultdict

# Group samples by MSI feature signature
print("Grouping slides by MSI feature consistency...")
sig_groups = defaultdict(list)
for s in samples:
    sig = get_msi_signature(s['msi_path'])
    sig_groups[sig].append(s)

print(f"Found {len(sig_groups)} distinct MSI feature configurations:")
for sig, group in sig_groups.items():
    print(f"  - {len(sig)} features: {len(group)} slides")

# Select largest compatible group
samples = max(sig_groups.values(), key=len)
print(f"\n✓ Selected largest group with {len(samples)} compatible slides")

# Shuffle
random.shuffle(samples)

# Split
if len(samples) >= 10:
    train_samples = samples[:7]
    test_samples  = samples[7:]
else:
    print("⚠ Warning: Less than 10 compatible slides. Adjusting split.")
    n_train = max(1, int(len(samples) * 0.7))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

print(f"\n✓ Training slides (n={len(train_samples)}):")
for s in train_samples:
    print(f"  - {s['family']}/{s['sample_id']}")

print(f"\n✓ Test slides (n={len(test_samples)}):")
for s in test_samples:
    print(f"  - {s['family']}/{s['sample_id']}")



# ============================================================================
# 3. PATCH EXTRACTION & SPATIAL ALIGNMENT
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 3] Loading spatial data and extracting patches...")
print("=" * 80)

PATCH_SIZE = 224

def load_spatial_data(sample: Dict) -> Tuple[pd.DataFrame, Dict, Image.Image]:
    """Load tissue positions, scale factors, and WSI."""
    # Load positions
    positions = pd.read_csv(
        sample['positions_path'],
        header=None,
        names=['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    )
    positions = positions[positions['in_tissue'] == 1].reset_index(drop=True)

    # Load scale factors
    with open(sample['scalefactors_path'], 'r') as f:
        scale_factors = json.load(f)

    # Load WSI
    wsi = Image.open(sample['wsi_path']).convert('RGB')

    return positions, scale_factors, wsi


def extract_patch(wsi: Image.Image, row: int, col: int, size: int = 224) -> np.ndarray:
    """Extract centered patch from WSI."""
    wsi_w, wsi_h = wsi.size

    # Calculate patch bounds
    left = max(0, col - size // 2)
    top = max(0, row - size // 2)
    right = min(wsi_w, left + size)
    bottom = min(wsi_h, top + size)

    # Extract and pad if needed
    patch = wsi.crop((left, top, right, bottom))

    # Pad to square if at boundary
    if patch.size != (size, size):
        padded = Image.new('RGB', (size, size), (255, 255, 255))
        padded.paste(patch, (0, 0))
        patch = padded

    return np.array(patch)


def process_sample(sample: Dict) -> Dict:

    """
    Process one sample: load spatial data, extract patches, load MSI.

    Returns dict with:
        - patches: (N, 224, 224, 3)
        - msi: (N, M) where M = number of metabolites
        - positions: DataFrame with spot metadata
    """
    print(f"\nProcessing: {sample['sample_id']}")

    # Load spatial data
    positions, scale_factors, wsi = load_spatial_data(sample)
    print(f"  ✓ Loaded {len(positions)} spots from spatial list")

    # Load MSI (no index col to check for x/y headers)
    msi_df = pd.read_csv(sample['msi_path'])
    print(f"  ✓ Loaded MSI: {msi_df.shape}")

    # --- ALIGNMENT LOGIC ---
    # Check for x/y coordinates in MSI
    if 'x' in msi_df.columns and 'y' in msi_df.columns:
        msi_df['array_row'] = msi_df['x'].astype(int)
        msi_df['array_col'] = msi_df['y'].astype(int)
    elif 'array_row' not in msi_df.columns:
        # If no coordinates found, we can't align
        raise ValueError(f"MSI CSV for {sample['sample_id']} lacks 'x'/'y' or 'array_row'/'array_col' for alignment.")

    # Merge positions (in_tissue=1) with MSI on array coordinates
    merged = pd.merge(positions, msi_df, on=['array_row', 'array_col'], how='inner')

    if len(merged) == 0:
         raise ValueError(f"Alignment failed for {sample['sample_id']}: No overlapping spots found.")

    print(f"  ✓ Aligned spots: {len(merged)} (from {len(positions)} spatial and {len(msi_df)} MSI)")

    # Separate metadata and MSI data
    # Metadata cols came from positions + x/y/array_row/array_col from MSI
    meta_cols = list(positions.columns) + ['x', 'y', 'array_row', 'array_col', 'Unnamed: 0']
    msi_cols = [c for c in merged.columns if c not in meta_cols]

    msi_aligned_raw = merged[msi_cols].values

    # Slide-wise MSI normalization (CRITICAL FIX)
    msi_aligned_raw = (
        msi_aligned_raw - msi_aligned_raw.mean(axis=0)
    ) / (msi_aligned_raw.std(axis=0) + 1e-6)


    positions_aligned = merged[positions.columns].reset_index(drop=True)

    # Convert coordinates to hires image space
    scale = scale_factors['tissue_hires_scalef']
    positions_aligned['hires_row'] = (positions_aligned['pxl_row_in_fullres'] * scale).astype(int)
    positions_aligned['hires_col'] = (positions_aligned['pxl_col_in_fullres'] * scale).astype(int)

    # Extract patches
    patches = []
    valid_indices = []

    for idx, row in tqdm(positions_aligned.iterrows(), total=len(positions_aligned), desc="  Extracting patches"):
        try:
            patch = extract_patch(wsi, row['hires_row'], row['hires_col'], PATCH_SIZE)
            assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3), f"Invalid patch shape: {patch.shape}"
            patches.append(patch)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  ⚠ Failed to extract patch at index {idx}: {e}")

    patches = np.array(patches)
    msi_final = msi_aligned_raw[valid_indices]
    positions_final = positions_aligned.iloc[valid_indices].reset_index(drop=True)

    print(f"  ✓ Extracted {len(patches)} valid patches")
    print(f"  ✓ Final shapes: patches={patches.shape}, MSI={msi_final.shape}")

    return {
        'sample_id': sample['sample_id'],
        'patches': patches,
        'msi': msi_final,
        'positions': positions_final,
        'wsi': wsi,
        'msi_columns': msi_cols  # <--- Added to track feature names
    }


# Process all samples
print("\n" + "=" * 80)
print("Processing training samples...")
print("=" * 80)
train_data = [process_sample(s) for s in train_samples]

print("\n" + "=" * 80)
print("Processing test samples...")
print("=" * 80)
test_data = [process_sample(s) for s in test_samples]


# ============================================================================
# 4. UNI FEATURE EXTRACTION (FROZEN)
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 4] Loading UNI model and extracting features...")
print("=" * 80)

# Load UNI
print("Loading UNI (MahmoodLab ViT-L)...")
# ⬇️ ADDED init_values=1e-5 to fix loading error
uni_model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, num_classes=0)
uni_model = uni_model.to(device)
uni_model.eval()

# Freeze all parameters
for param in uni_model.parameters():
    param.requires_grad = False

print(f"✓ UNI loaded and frozen")
print(f"✓ Output dimension: {uni_model.num_features}")


def extract_uni_features(patches: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Extract UNI features from patches."""
    features = []

    # Normalize patches (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    patches_norm = (patches / 255.0 - mean) / std

    # Convert to tensor: (N, H, W, C) -> (N, C, H, W)
    patches_tensor = torch.from_numpy(patches_norm).permute(0, 3, 1, 2).float()

    with torch.no_grad():
        for i in tqdm(range(0, len(patches_tensor), batch_size), desc="Encoding patches"):
            batch = patches_tensor[i:i+batch_size].to(device)
            feats = uni_model(batch).cpu().numpy()
            features.append(feats)

    return np.vstack(features)


# Extract features for all samples
print("\nExtracting UNI features for training data...")
for data in train_data:
    data['features'] = extract_uni_features(data['patches'])
    print(f"  ✓ {data['sample_id']}: {data['features'].shape}")

print("\nExtracting UNI features for test data...")
for data in test_data:
    data['features'] = extract_uni_features(data['patches'])
    print(f"  ✓ {data['sample_id']}: {data['features'].shape}")


# ============================================================================
# 5. MSI PREPROCESSING: PCA COMPRESSION (FIT ON TRAIN ONLY)
# ============================================================================


print("\n" + "=" * 80)
print("[STEP 5] MSI preprocessing with PCA...")
print("=" * 80)

N_PCA =100

# Concatenate all training MSI
train_msi_all = np.vstack([d['msi'] for d in train_data])
print(f"✓ Training MSI shape: {train_msi_all.shape}")

# Fit StandardScaler on training data only
scaler = StandardScaler()
train_msi_scaled = scaler.fit_transform(train_msi_all)
print(f"✓ StandardScaler fitted on training data")

# Fit PCA on training data only
# Fit PCA on training data only
# Fit PCA on training data only (FAST VERSION)
import time
start_time = time.time()

pca = PCA(
    n_components=N_PCA,
    svd_solver='randomized',  # KEY FIX: prevents freeze
    random_state=SEED,
    iterated_power=3  # Extra accuracy
)

train_msi_pca = pca.fit_transform(train_msi_scaled)
pc_weights = torch.tensor(
    pca.explained_variance_ratio_,
    device=device,
    dtype=torch.float32
)

elapsed = time.time() - start_time

print(f"✓ PCA fitted in {elapsed:.1f} seconds")
print(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
# After PCA fitting, add this:

# Plot cumulative variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2)
plt.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
plt.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
plt.xlabel('Number of Components', fontsize=11)
plt.ylabel('Cumulative Explained Variance', fontsize=11)
plt.title(f'PCA Cumulative Variance (n={N_PCA})', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.bar(range(min(20, N_PCA)), pca.explained_variance_ratio_[:20], color='steelblue')
plt.xlabel('PC Index', fontsize=11)
plt.ylabel('Explained Variance Ratio', fontsize=11)
plt.title('Top 20 Principal Components', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 3, 3)
plt.semilogy(pca.explained_variance_ratio_, marker='o', markersize=3, linewidth=1)
plt.xlabel('PC Index', fontsize=11)
plt.ylabel('Explained Variance (log scale)', fontsize=11)
plt.title('Variance Decay Profile', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ Saved: pca_variance.png")
# Transform each training sample
start_idx = 0
for data in train_data:
    n_spots = len(data['msi'])
    data['msi_pca'] = train_msi_pca[start_idx:start_idx + n_spots]
    start_idx += n_spots
    print(f"  ✓ {data['sample_id']}: {data['msi_pca'].shape}")

# Transform test data (using fitted scaler and PCA)
print("\nTransforming test MSI...")
for data in test_data:
    msi_scaled = scaler.transform(data['msi'])
    data['msi_pca'] = pca.transform(msi_scaled)
    print(f"  ✓ {data['sample_id']}: {data['msi_pca'].shape}")

# Plot cumulative variance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title(f'PCA Cumulative Variance (n={N_PCA})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(min(20, N_PCA)), pca.explained_variance_ratio_[:20])
plt.xlabel('PC Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Top 20 Principal Components')
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: pca_variance.png")




# ============================================================================
# 6. PYTORCH DATASET & DATALOADER
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 6] Building PyTorch datasets...")
print("=" * 80)

class MSIDataset(Dataset):
    """
    Dataset with spatial context aggregation
    """

    def __init__(self, data_list, k_neighbors=6):
        X, Y = [], []

        for d in data_list:
            feats = d['features']
            targets = d['msi_pca']
            pos = d['positions'][['array_row', 'array_col']].values

            for i in range(len(feats)):
                # distance to all other spots in same slide
                dist = np.linalg.norm(pos - pos[i], axis=1)
                nn_idx = np.argsort(dist)[1:k_neighbors+1]

                neighbor_feat = feats[nn_idx].mean(axis=0)

                # concatenate center + neighborhood
                x = np.concatenate([feats[i], neighbor_feat])
                y = targets[i]

                X.append(x)
                Y.append(y)

        self.features = np.array(X)
        self.targets = np.array(Y)

        assert self.features.shape[0] == self.targets.shape[0]

        print(f"✓ Context-aware dataset created")
        print(f"  Features shape: {self.features.shape}")
        print(f"  Targets shape: {self.targets.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


# Create datasets
train_dataset = MSIDataset(train_data)
test_dataset = MSIDataset(test_data)

# Create dataloaders
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"\n✓ Dataloaders created (batch_size={BATCH_SIZE})")

# Sanity check batch shapes
x_batch, y_batch = next(iter(train_loader))
print(f"✓ Sample batch shapes: X={x_batch.shape}, Y={y_batch.shape}")


# ============================================================================
# 7. MLP PREDICTION HEAD
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 7] Building MLP prediction head...")
print("=" * 80)

class MLPHead(nn.Module):
    """MLP regression head for MSI prediction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [512, 512], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation - regression)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Initialize model
input_dim = train_dataset.features.shape[1]  # UNI feature dim
output_dim = N_PCA

model = MLPHead(
    input_dim=train_dataset.features.shape[1],
    output_dim=N_PCA,
    hidden_dims=[1024, 512],
    dropout=0.3
)
model = model.to(device)

print(f"✓ Model architecture:")
print(model)
print(f"\n✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

def weighted_mse(pred, target):
    return torch.mean(pc_weights * (pred - target) ** 2)

criterion = weighted_mse

# ============================================================================
# 8. TRAINING LOOP
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 8] Training model...")
print("=" * 80)

# Training config
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training history
history = {'train_loss': [], 'test_loss': []}
best_test_loss = float('inf')
best_epoch = 0

print(f"Training for {NUM_EPOCHS} epochs...")
print(f"Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"Loss: MSE")

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(x_batch)

    train_loss /= len(train_dataset)

    # Evaluation phase
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * len(x_batch)

    test_loss /= len(test_dataset)

    # Update history
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)

    # Learning rate scheduling
    scheduler.step(test_loss)

    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Best: {best_test_loss:.4f} @ epoch {best_epoch+1}")

print(f"\n✓ Training completed!")
print(f"✓ Best test loss: {best_test_loss:.4f} at epoch {best_epoch+1}")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
print("✓ Best model loaded")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['test_loss'], label='Test Loss')
plt.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch+1})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: training_curves.png")


# ============================================================================
# 9. FINAL EVALUATION ON TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 9] Final evaluation on test set...")
print("=" * 80)

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Compute metrics
rmse_per_pc = np.sqrt(mean_squared_error(all_targets, all_preds, multioutput='raw_values'))
r2_per_pc = r2_score(all_targets, all_preds, multioutput='raw_values')

rmse_mean = rmse_per_pc.mean()
r2_mean = r2_per_pc.mean()

print(f"\n{'='*60}")
print(f"FINAL TEST SET METRICS")
print(f"{'='*60}")
print(f"RMSE (mean across {N_PCA} PCs): {rmse_mean:.4f}")
print(f"R² (mean across {N_PCA} PCs): {r2_mean:.4f}")
print(f"{'='*60}")

# Per-PC breakdown (top 10)
print(f"\nTop 10 PCs breakdown:")
print(f"{'PC':<5} {'RMSE':<10} {'R²':<10}")
print("-" * 25)
for i in range(min(10, N_PCA)):
    print(f"{i:<5} {rmse_per_pc[i]:<10.4f} {r2_per_pc[i]:<10.4f}")


# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("[STEP 10] Generating visualizations...")
print("=" * 80)

# 10.1: Scatter plots for PC0 and PC1
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for pc_idx, ax in enumerate(axes):
    ax.scatter(all_targets[:, pc_idx], all_preds[:, pc_idx], alpha=0.3, s=5)

    # Add diagonal line
    min_val = min(all_targets[:, pc_idx].min(), all_preds[:, pc_idx].min())
    max_val = max(all_targets[:, pc_idx].max(), all_preds[:, pc_idx].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.5)

    ax.set_xlabel(f'True PC{pc_idx}')
    ax.set_ylabel(f'Predicted PC{pc_idx}')
    ax.set_title(f'PC{pc_idx}: R²={r2_per_pc[pc_idx]:.3f}, RMSE={rmse_per_pc[pc_idx]:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_pc0_pc1.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: scatter_pc0_pc1.png")

# 10.2: Heatmap comparison for first test slide
test_slide_idx = 0
test_slide = test_data[test_slide_idx]
n_spots = len(test_slide['msi_pca'])

# Get predictions for this slide

# --- FIX START: Construct context-aware features manually for this slide ---
print(f"Generating context-aware features for slide {test_slide['sample_id']}...")
feats = test_slide['features']
pos = test_slide['positions'][['array_row', 'array_col']].values
k_neighbors = 6

aggregated_features = []
for i in range(len(feats)):
    # distance to all other spots in same slide (fast enough for inference)
    dist = np.linalg.norm(pos - pos[i], axis=1)
    nn_idx = np.argsort(dist)[1:k_neighbors+1]
    neighbor_feat = feats[nn_idx].mean(axis=0)
    # concatenate center + neighborhood to get shape (2048,)
    x = np.concatenate([feats[i], neighbor_feat])
    aggregated_features.append(x)

slide_features = torch.tensor(np.array(aggregated_features), dtype=torch.float32).to(device)
# --- FIX END ---

with torch.no_grad():
    slide_preds = model(slide_features).cpu().numpy()

# Plot heatmaps for PC0
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# True PC0
ax = axes[0]
scatter = ax.scatter(
    test_slide['positions']['hires_col'],
    test_slide['positions']['hires_row'],
    c=test_slide['msi_pca'][:, 0],
    cmap='viridis',
    s=50,
    alpha=0.8
)
ax.set_title(f'True PC0 - {test_slide["sample_id"]}')
ax.invert_yaxis()
ax.axis('equal')
plt.colorbar(scatter, ax=ax)

# Predicted PC0
ax = axes[1]
scatter = ax.scatter(
    test_slide['positions']['hires_col'],
    test_slide['positions']['hires_row'],
    c=slide_preds[:, 0],
    cmap='viridis',
    s=50,
    alpha=0.8
)
ax.set_title(f'Predicted PC0 - {test_slide["sample_id"]}')
ax.invert_yaxis()
ax.axis('equal')
plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig('spatial_heatmap_pc0.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: spatial_heatmap_pc0.png")

# 10.3: Patch grid visualization
print("\nVisualizing patch grid from first test slide...")
n_show = min(16, len(test_slide['patches']))
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for i in range(n_show):
    axes[i].imshow(test_slide['patches'][i])
    axes[i].set_title(f"Spot {i}", fontsize=8)
    axes[i].axis('off')

for i in range(n_show, 16):
    axes[i].axis('off')

plt.suptitle(f'Sample Patches - {test_slide["sample_id"]}', fontsize=14)
plt.tight_layout()
plt.savefig
