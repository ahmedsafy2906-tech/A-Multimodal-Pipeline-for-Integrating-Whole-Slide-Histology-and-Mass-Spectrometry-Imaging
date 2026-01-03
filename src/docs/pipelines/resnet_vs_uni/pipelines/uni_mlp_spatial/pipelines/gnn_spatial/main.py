
import os
import json
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Using device: {device}")

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("[INFO] Imports completed successfully!")

################################################################################
# HELPER FUNCTIONS
################################################################################

def discover_samples(root: str):
    """Discover samples matching old code's structure."""
    print(f"\n{'='*80}")
    print("DISCOVERING SAMPLES")
    print(f"{'='*80}")

    samples = []

    for family_dir in Path(root).glob("*"):
        if not family_dir.is_dir() or family_dir.name.startswith('.'):
            continue

        for sample_dir in family_dir.glob("*"):
            if not sample_dir.is_dir() or sample_dir.name.startswith('.'):
                continue

            output_dir = sample_dir / "output_data"
            if not output_dir.exists():
                continue

            rna_dirs = list(output_dir.glob("*_RNA"))
            msi_dirs = list(output_dir.glob("*_MSI"))

            if not rna_dirs or not msi_dirs:
                continue

            rna_dir = rna_dirs[0]
            msi_dir = msi_dirs[0]

            spatial_dir = rna_dir / "outs" / "spatial"
            wsi_path = spatial_dir / "tissue_hires_image.png"
            positions_path = spatial_dir / "tissue_positions_list.csv"
            scalefactors_path = spatial_dir / "scalefactors_json.json"

            msi_csvs = list(msi_dir.glob("*.csv"))

            if not all([wsi_path.exists(), positions_path.exists(),
                       scalefactors_path.exists(), len(msi_csvs) > 0]):
                print(f"  ⚠ Skipping {sample_dir.name}: missing files")
                continue

            samples.append({
                "sample_id": sample_dir.name,
                "family": family_dir.name,
                "wsi_path": str(wsi_path),
                "positions_path": str(positions_path),
                "scalefactors_path": str(scalefactors_path),
                "msi_path": str(msi_csvs[0])
            })
            print(f"  ✓ Found: {family_dir.name}/{sample_dir.name}")

    print(f"\n✓ Total valid samples: {len(samples)}")
    return samples


def get_msi_signature(msi_path: str):
    """Get MSI feature signature."""
    cols = pd.read_csv(msi_path, nrows=0).columns.tolist()
    meta_cols = {'x', 'y', 'array_row', 'array_col', 'Unnamed: 0'}
    return tuple(sorted([c for c in cols if c not in meta_cols]))


def load_spatial_data(sample):
    """Load spatial data."""
    positions = pd.read_csv(
        sample['positions_path'], header=None,
        names=['barcode', 'in_tissue', 'array_row', 'array_col',
               'pxl_row_in_fullres', 'pxl_col_in_fullres']
    )
    positions = positions[positions['in_tissue'] == 1].reset_index(drop=True)

    with open(sample['scalefactors_path'], 'r') as f:
        scale_factors = json.load(f)

    wsi = Image.open(sample['wsi_path']).convert('RGB')
    return positions, scale_factors, wsi


def extract_patch(wsi: Image.Image, row: int, col: int, size: int = 224):
    """Extract patch from WSI."""
    wsi_w, wsi_h = wsi.size
    left = max(0, col - size // 2)
    top = max(0, row - size // 2)
    right = min(wsi_w, left + size)
    bottom = min(wsi_h, top + size)

    patch = wsi.crop((left, top, right, bottom))

    if patch.size != (size, size):
        padded = Image.new('RGB', (size, size), (255, 255, 255))
        padded.paste(patch, (0, 0))
        patch = padded

    return np.array(patch)


def extract_resnet50_features(patches, batch_size=32):
    """Extract ResNet50 features."""
    print(f"  [INFO] Extracting ResNet50 features from {len(patches)} patches...")

    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
    resnet50 = resnet50.to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="    Encoding"):
            batch = patches[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(p) for p in batch]).to(device)
            feat = resnet50(batch_tensor).squeeze(-1).squeeze(-1)
            features.append(feat.cpu().numpy())

    return np.vstack(features)


def process_sample(sample):
    """Process one sample - FIXED ALIGNMENT."""
    print(f"\n  Processing: {sample['sample_id']}")

    positions, scale_factors, wsi = load_spatial_data(sample)
    print(f"    ✓ Loaded {len(positions)} spatial spots")

    msi_df = pd.read_csv(sample['msi_path'])
    print(f"    ✓ Loaded MSI: {msi_df.shape}")

    # FIXED ALIGNMENT
    if 'x' in msi_df.columns and 'y' in msi_df.columns:
        msi_df['array_row'] = msi_df['x'].astype(int)
        msi_df['array_col'] = msi_df['y'].astype(int)

    merged = pd.merge(positions, msi_df, on=['array_row', 'array_col'], how='inner')

    if len(merged) == 0:
        raise ValueError(f"No alignment for {sample['sample_id']}!")

    print(f"    ✓ Aligned {len(merged)} spots")

    meta_cols = list(positions.columns) + ['x', 'y', 'array_row', 'array_col', 'Unnamed: 0']
    msi_cols = [c for c in merged.columns if c not in meta_cols]

    msi_aligned = merged[msi_cols].values
    positions_aligned = merged[positions.columns].reset_index(drop=True)

    scale = scale_factors['tissue_hires_scalef']
    positions_aligned['hires_row'] = (positions_aligned['pxl_row_in_fullres'] * scale).astype(int)
    positions_aligned['hires_col'] = (positions_aligned['pxl_col_in_fullres'] * scale).astype(int)

    patches = []
    valid_indices = []

    for idx, row in positions_aligned.iterrows():
        try:
            patch = extract_patch(wsi, row['hires_row'], row['hires_col'], 224)
            patches.append(patch)
            valid_indices.append(idx)
        except:
            pass

    patches = np.array(patches)
    msi_final = msi_aligned[valid_indices]
    positions_final = positions_aligned.iloc[valid_indices].reset_index(drop=True)

    print(f"    ✓ Extracted {len(patches)} patches")

    wsi_features = extract_resnet50_features(patches, 32)

    return {
        'sample_id': sample['sample_id'],
        'wsi': np.array(wsi),
        'patches': patches,
        'wsi_features': wsi_features,
        'msi_raw': msi_final,
        'positions': positions_final,
        'msi_columns': msi_cols,
        'coordinates': positions_final[['hires_col', 'hires_row']].values
    }


def apply_pca_to_data(train_data, test_data, n_components=200):
    """Apply PCA - FIT ON TRAIN ONLY."""
    print(f"\n{'='*80}")
    print("APPLYING PCA TO MSI DATA")
    print(f"{'='*80}")

    train_msi_all = np.vstack([d['msi_raw'] for d in train_data])
    print(f"✓ Training MSI shape: {train_msi_all.shape}")

    scaler = StandardScaler()
    train_msi_scaled = scaler.fit_transform(train_msi_all)

    pca = PCA(n_components=min(n_components, train_msi_scaled.shape[0]-1,
                               train_msi_scaled.shape[1]),
              svd_solver='randomized', random_state=SEED)
    train_msi_pca = pca.fit_transform(train_msi_scaled)

    print(f"✓ PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    start_idx = 0
    for data in train_data:
        n_spots = len(data['msi_raw'])
        data['msi_pca'] = train_msi_pca[start_idx:start_idx + n_spots]
        start_idx += n_spots

    for data in test_data:
        msi_scaled = scaler.transform(data['msi_raw'])
        data['msi_pca'] = pca.transform(msi_scaled)

    return scaler, pca


################################################################################
# SPATIAL GRAPH & GNN MODEL
################################################################################

class SpatialGraphBuilder:
    """Build k-NN spatial graphs."""
    def __init__(self, k_neighbors=6, self_loop=True):
        self.k_neighbors = k_neighbors
        self.self_loop = self_loop

    def build_graph(self, coordinates):
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1,
                                algorithm='ball_tree').fit(coordinates)
        distances, indices = nbrs.kneighbors(coordinates)

        edge_list = []
        for i in range(len(coordinates)):
            for j in indices[i, 1:]:
                edge_list.append([i, j])

        edge_index = torch.LongTensor(edge_list).t().contiguous()

        if self.self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=len(coordinates))

        return edge_index


class WSI2MSI_GNN(nn.Module):
    """GNN for WSI-to-MSI prediction."""
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=200,
                 num_gnn_layers=3, num_attention_heads=4, dropout=0.2):
        super().__init__()

        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                GATv2Conv(hidden_dim, hidden_dim, heads=num_attention_heads,
                         dropout=dropout, concat=False, add_self_loops=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x, edge_index, batch=None):
        h = self.input_encoder(x)

        for gnn_layer, batch_norm in zip(self.gnn_layers, self.batch_norms):
            h_residual = h
            h = gnn_layer(h, edge_index)
            h = batch_norm(h)
            h = torch.relu(h)
            h = h + h_residual
            h = self.dropout(h)

        return self.readout_mlp(h)


class CompositeLoss(nn.Module):
    """Composite loss: MSE + Pearson + Cosine."""
    def __init__(self, mse_weight=1.0, pearson_weight=0.5, cosine_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.cosine_weight = cosine_weight
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def pearson_correlation_loss(self, pred, target):
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        covariance = (pred_centered * target_centered).sum(dim=0)
        pred_std = pred_centered.pow(2).sum(dim=0).sqrt()
        target_std = target_centered.pow(2).sum(dim=0).sqrt()
        correlation = covariance / (pred_std * target_std + 1e-8)
        return 1.0 - correlation.mean()

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        pearson = self.pearson_correlation_loss(pred, target)
        target_labels = torch.ones(pred.size(0), device=pred.device)
        cosine = self.cosine_loss(pred, target, target_labels)
        total = self.mse_weight * mse + self.pearson_weight * pearson + self.cosine_weight * cosine
        return total, {'total': total.item(), 'mse': mse.item(),
                       'pearson': pearson.item(), 'cosine': cosine.item()}


def train_gnn_model(full_graph, input_dim, output_dim, hidden_dim=512,
                    num_gnn_layers=3, learning_rate=1e-4, epochs=50,
                    save_path='./model.pth'):
    """Train GNN."""
    print(f"\n{'='*80}")
    print("TRAINING GNN MODEL")
    print(f"{'='*80}")

    model = WSI2MSI_GNN(input_dim, hidden_dim, output_dim, num_gnn_layers).to(device)
    criterion = CompositeLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    full_graph = full_graph.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"Training for {epochs} epochs...")
    print(f"Train nodes: {full_graph.train_mask.sum().item()}")
    print(f"Val nodes: {full_graph.val_mask.sum().item()}\n")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(full_graph.x, full_graph.edge_index)
        train_loss, train_dict = criterion(predictions[full_graph.train_mask],
                                          full_graph.y[full_graph.train_mask])
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_dict['total'])

        model.eval()
        with torch.no_grad():
            predictions = model(full_graph.x, full_graph.edge_index)
            val_loss, val_dict = criterion(predictions[full_graph.val_mask],
                                          full_graph.y[full_graph.val_mask])
        val_losses.append(val_dict['total'])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_dict['total']:.6f} | Val: {val_dict['total']:.6f}")

        if val_dict['total'] < best_val_loss:
            best_val_loss = val_dict['total']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"\n✓ Training completed!")
    print(f"✓ Best val loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses


def evaluate_with_comprehensive_metrics(model, wsi_features, msi_features,
                                        coordinates, sample_id, k_neighbors=6):
    """Evaluate with R², RMSE, F1."""
    model.eval()

    graph_builder = SpatialGraphBuilder(k_neighbors, self_loop=True)
    edge_index = graph_builder.build_graph(coordinates)

    sample_graph = Data(
        x=torch.FloatTensor(wsi_features),
        y=torch.FloatTensor(msi_features),
        edge_index=edge_index
    ).to(device)

    with torch.no_grad():
        predictions = model(sample_graph.x, sample_graph.edge_index).cpu().numpy()

    targets = msi_features

    # R² SCORE
    r2_overall = r2_score(targets, predictions, multioutput='variance_weighted')
    r2_per_pc = r2_score(targets, predictions, multioutput='raw_values')

    # RMSE
    rmse_overall = np.sqrt(mean_squared_error(targets, predictions))
    rmse_per_pc = np.sqrt(mean_squared_error(targets, predictions, multioutput='raw_values'))

    # F1 SCORE
    targets_binary = (targets > np.median(targets, axis=0)).astype(int)
    preds_binary = (predictions > np.median(predictions, axis=0)).astype(int)

    f1_per_pc = []
    for i in range(targets.shape[1]):
        try:
            f1 = f1_score(targets_binary[:, i], preds_binary[:, i],
                         average='binary', zero_division=0)
            f1_per_pc.append(f1)
        except:
            f1_per_pc.append(0.0)

    f1_macro = np.mean(f1_per_pc)

    # PEARSON
    correlations = []
    for i in range(targets.shape[1]):
        if np.std(targets[:, i]) > 0 and np.std(predictions[:, i]) > 0:
            corr, _ = pearsonr(targets[:, i], predictions[:, i])
            correlations.append(corr if not np.isnan(corr) else 0.0)
        else:
            correlations.append(0.0)

    metrics = {
        'sample_id': sample_id,
        'r2_overall': r2_overall,
        'r2_per_pc': r2_per_pc,
        'r2_mean': np.mean(r2_per_pc),
        'rmse_overall': rmse_overall,
        'rmse_per_pc': rmse_per_pc,
        'rmse_mean': np.mean(rmse_per_pc),
        'f1_macro': f1_macro,
        'f1_per_pc': f1_per_pc,
        'pearson_mean': np.mean(correlations),
        'correlations': correlations,
        'n_spots': len(wsi_features)
    }

    print(f"\n[EVAL] {sample_id}:")
    print(f"  R² Overall: {r2_overall:.4f}")
    print(f"  RMSE Overall: {rmse_overall:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  Pearson Mean: {metrics['pearson_mean']:.4f}")

    return predictions, metrics


def create_visualizations(sample_id, wsi_image, coordinates, msi_features,
                         predictions, metrics, output_dir):
    """Generate visualizations."""
    viz_dir = os.path.join(output_dir, sample_id)
    os.makedirs(viz_dir, exist_ok=True)

    # Metrics summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].bar(range(len(metrics['r2_per_pc'])), metrics['r2_per_pc'], color='steelblue', alpha=0.7)
    axes[0, 0].axhline(metrics['r2_mean'], color='red', linestyle='--', label=f"Mean: {metrics['r2_mean']:.4f}")
    axes[0, 0].set_title('R² Score per PC', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(range(len(metrics['rmse_per_pc'])), metrics['rmse_per_pc'], color='coral', alpha=0.7)
    axes[0, 1].axhline(metrics['rmse_mean'], color='red', linestyle='--', label=f"Mean: {metrics['rmse_mean']:.4f}")
    axes[0, 1].set_title('RMSE per PC', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(range(len(metrics['f1_per_pc'])), metrics['f1_per_pc'], color='mediumseagreen', alpha=0.7)
    axes[1, 0].axhline(metrics['f1_macro'], color='red', linestyle='--', label=f"Macro: {metrics['f1_macro']:.4f}")
    axes[1, 0].set_title('F1 Score per PC', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(range(len(metrics['correlations'])), metrics['correlations'], color='orchid', alpha=0.7)
    axes[1, 1].axhline(metrics['pearson_mean'], color='red', linestyle='--', label=f"Mean: {metrics['pearson_mean']:.4f}")
    axes[1, 1].set_title('Pearson Correlation per PC', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Comprehensive Metrics - {sample_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '01_metrics.png'), dpi=300)
    plt.close()

    # Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    vmin = min(msi_features[:, 0].min(), predictions[:, 0].min())
    vmax = max(msi_features[:, 0].max(), predictions[:, 0].max())

    axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=msi_features[:, 0],
                    cmap='RdBu_r', s=60, vmin=vmin, vmax=vmax)
    axes[0].set_title('True MSI PC1', fontweight='bold')
    axes[0].invert_yaxis()

    axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=predictions[:, 0],
                    cmap='RdBu_r', s=60, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Predicted PC1 (R²={metrics["r2_per_pc"][0]:.3f})', fontweight='bold')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '02_heatmap_pc1.png'), dpi=300)
    plt.close()

    print(f"  ✓ Visualizations saved to: {viz_dir}")


################################################################################
# MAIN PIPELINE
################################################################################

def main_fixed_pipeline(data_root, output_dir, k_neighbors=6, hidden_dim=512,
                        num_gnn_layers=3, learning_rate=1e-4, epochs=50, n_pca=100):
    """Main pipeline execution."""
    print(f"\n{'='*80}")
    print("FIXED WSI-TO-MSI GNN PIPELINE")
    print(f"{'='*80}")

    os.makedirs(output_dir, exist_ok=True)

    # Discover samples
    samples = discover_samples(data_root)

    if len(samples) == 0:
        raise ValueError("No samples found!")

    # Group by MSI signature
    sig_groups = defaultdict(list)
    for s in samples:
        sig = get_msi_signature(s['msi_path'])
        sig_groups[sig].append(s)

    samples = max(sig_groups.values(), key=len)
    print(f"\n✓ Selected {len(samples)} compatible slides")

    # Split train/test
    import random
    random.seed(SEED)
    random.shuffle(samples)

    n_train = max(1, int(len(samples) * 0.7))
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    print(f"✓ Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Process samples
    print(f"\n{'='*80}")
    print("PROCESSING SAMPLES")
    print(f"{'='*80}")

    train_data = [process_sample(s) for s in train_samples]
    test_data = [process_sample(s) for s in test_samples]

    # Apply PCA
    scaler, pca = apply_pca_to_data(train_data, test_data, n_pca)

    # Build graph
    X_train = np.vstack([d['wsi_features'] for d in train_data])
    y_train = np.vstack([d['msi_pca'] for d in train_data])
    coords_train = np.vstack([d['coordinates'] for d in train_data])

    print(f"\n{'='*80}")
    print("BUILDING SPATIAL GRAPH")
    print(f"{'='*80}")
    print(f"Total samples: {len(X_train)}")
    print(f"WSI features: {X_train.shape}")
    print(f"MSI features: {y_train.shape}")

    graph_builder = SpatialGraphBuilder(k_neighbors, self_loop=True)
    edge_index = graph_builder.build_graph(coords_train)

    n_samples = len(X_train)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)

    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    train_mask[indices[:split_idx]] = True
    val_mask[indices[split_idx:]] = True

    full_graph = Data(
        x=torch.FloatTensor(X_train),
        y=torch.FloatTensor(y_train),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask
    )

    print(f"✓ Graph: {full_graph.x.size(0)} nodes, {full_graph.edge_index.size(1)} edges")

    # Train model
    model, train_losses, val_losses = train_gnn_model(
        full_graph, X_train.shape[1], y_train.shape[1],
        hidden_dim, num_gnn_layers, learning_rate, epochs,
        os.path.join(output_dir, 'model.pth')
    )

    # Evaluate
    print(f"\n{'='*80}")
    print("EVALUATION")
    print(f"{'='*80}")

    all_metrics = []

    for data in test_data:
        predictions, metrics = evaluate_with_comprehensive_metrics(
            model, data['wsi_features'], data['msi_pca'],
            data['coordinates'], data['sample_id'], k_neighbors
        )

        all_metrics.append(metrics)

        # Save predictions
        pred_path = os.path.join(output_dir, data['sample_id'], 'predictions.npy')
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        np.save(pred_path, predictions)

        # Visualize
        create_visualizations(
            data['sample_id'], data['wsi'], data['coordinates'],
            data['msi_pca'], predictions, metrics, output_dir
        )

    # Save summary
    summary_data = []
    for m in all_metrics:
        summary_data.append({
            'Sample': m['sample_id'],
            'R²_Overall': f"{m['r2_overall']:.4f}",
            'R²_Mean': f"{m['r2_mean']:.4f}",
            'RMSE_Overall': f"{m['rmse_overall']:.4f}",
            'RMSE_Mean': f"{m['rmse_mean']:.4f}",
            'F1_Macro': f"{m['f1_macro']:.4f}",
            'Pearson_Mean': f"{m['pearson_mean']:.4f}",
            'N_Spots': m['n_spots']
        })

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}")
    print(f"\n✓ All outputs saved to: {output_dir}")
    print(f"✓ Summary: {csv_path}")
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}\n")


################################################################################
# CONFIGURATION & EXECUTION
################################################################################

CONFIG = {
    'data_root': "/content/drive/MyDrive/sma/sma",
    'output_dir': "/content/drive/MyDrive/sma/outputs_fixed_gnn",
    'k_neighbors': 6,
    'hidden_dim': 512,
    'num_gnn_layers': 3,
    'learning_rate': 1e-4,
    'epochs': 50,
    'n_pca': 200,
}

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    for key, value in CONFIG.items():
        print(f"  {key:.<30} {value}")
    print("="*80 + "\n")

    print("[INFO] Checking data path...")
    if not os.path.exists(CONFIG['data_root']):
        print(f"\n ERROR: Data path not found!")
        print(f"   Path: {CONFIG['data_root']}")
        print(f"\n   Your available directories:")
        !find /content/drive -name "sma" -type d 2>/dev/null | head -10
    else:
        print(f"✓ Data path exists!")
        print(f"✓ Found {len(os.listdir(CONFIG['data_root']))} items\n")

        try:
            main_fixed_pipeline(**CONFIG)
        except Exception as e:
            print(f"\n ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

print("\n[INFO] Code loaded successfully! Run the cell to execute.")
