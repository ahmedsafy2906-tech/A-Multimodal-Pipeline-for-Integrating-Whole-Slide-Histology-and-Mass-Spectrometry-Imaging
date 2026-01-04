
from google.colab import drive
drive.mount('/content/drive')

DATA_ROOT = "/content/drive/MyDrive/sma/sma"

import torch, numpy as np, pandas as pd
from pathlib import Path
import cv2, json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import timm
from huggingface_hub import login
import matplotlib.pyplot as plt
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
N_PCA = 100
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# 1. ENCODERS + TRANSFORMS
resnet50 = timm.create_model('resnet50', pretrained=True, num_classes=0).eval().to(device)
resnet_transform = create_transform(**resolve_data_config({}, model=resnet50))
uni = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5).eval().to(device)
uni_transform = create_transform(**resolve_data_config({}, model=uni))
print("Encoders + transforms ready")

# 2. DATA DISCOVERY
def discover_samples(base_path):
    samples = []
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"Dataset not found at: {base}")

    for family_dir in base.iterdir():
        if not family_dir.is_dir() or family_dir.name.startswith('.'):
            continue

        for rep_dir in family_dir.iterdir():
            if not rep_dir.is_dir() or rep_dir.name.startswith('.'):
                continue

            out_dir = rep_dir / "output_data"
            if not out_dir.exists():
                continue

            msi_dirs = [d for d in out_dir.iterdir()
                        if d.is_dir() and not d.name.startswith('.') and d.name.endswith("_MSI")]
            rna_dirs = [d for d in out_dir.iterdir()
                        if d.is_dir() and not d.name.startswith('.') and d.name.endswith("_RNA")]
            if not msi_dirs or not rna_dirs:
                continue

            msi_csvs = list(msi_dirs[0].glob("*.csv"))
            spatial = rna_dirs[0] / "outs" / "spatial"
            pos_file = spatial / "tissue_positions_list.csv"
            tissue_png = spatial / "tissue_hires_image.png"

            if msi_csvs and pos_file.exists() and tissue_png.exists():
                samples.append({
                    "sample_id": rep_dir.name,
                    "family": family_dir.name,
                    "msi_csv": msi_csvs[0],
                    "positions_csv": pos_file,
                    "tissue_png": tissue_png,
                    "scalefactors": spatial / "scalefactors_json.json",
                })
    return samples

all_samples = discover_samples(DATA_ROOT)
print(f"Found {len(all_samples)} valid samples")
for s in all_samples:
    print(f"  - {s['sample_id']}")

# 3. FEATURE EXTRACTION HELPERS
def extract_features_batched(patches, model, transform, batch_size=BATCH_SIZE):
    model.eval()
    features = []

    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        batch_tensors = []
        for patch in batch_patches:
            patch_np = patch.astype("float32") / 255.0
            patch_tensor = torch.from_numpy(patch_np).permute(2, 0, 1)
            patch_tensor = transform(patch_tensor)
            batch_tensors.append(patch_tensor.unsqueeze(0))

        batch_tensor = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            feats = model(batch_tensor).cpu().numpy()
        features.append(feats)
        del batch_tensor, batch_tensors

    return np.vstack(features)

def extract_sample_data(sample, patch_size=224):
    """Extract WSI patches and MSI spectra for one slide."""
    msi_df = pd.read_csv(sample["msi_csv"])
    assert "x" in msi_df.columns and "y" in msi_df.columns
    msi_df["array_row"] = msi_df["x"].astype(int)
    msi_df["array_col"] = msi_df["y"].astype(int)

    pos_df = pd.read_csv(
        sample["positions_csv"],
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "px_lowres", "py_lowres"],
    )
    pos_df = pos_df[pos_df["in_tissue"] == 1]

    merged = msi_df.merge(
        pos_df,
        on=["array_row", "array_col"],
        how="inner",
        suffixes=("_msi", "_pos"),
    )
    assert len(merged) > 0, f"No overlapping spots for {sample['sample_id']}"

    with open(sample["scalefactors"], "r") as f:
        sf = json.load(f)
    hires_sc = sf["tissue_hires_scalef"]

    tissue = cv2.cvtColor(cv2.imread(str(sample["tissue_png"])), cv2.COLOR_BGR2RGB)
    assert tissue is not None
    h, w = tissue.shape[:2]

    drop_cols = ["x", "y", "barcode", "in_tissue",
                 "array_row", "array_col", "px_lowres", "py_lowres"]
    mz_cols = [c for c in merged.columns if c not in drop_cols]
    msi_mat_full = merged[mz_cols].values.astype(float)

    px_hires = (merged["px_lowres"].values * hires_sc).astype(float)
    py_hires = (merged["py_lowres"].values * hires_sc).astype(float)

    patches = []
    valid_idx = []
    for i, (px, py) in enumerate(zip(px_hires, py_hires)):
        px, py = int(round(px)), int(round(py))
        x1 = max(0, px - patch_size // 2)
        y1 = max(0, py - patch_size // 2)
        x2 = min(w, px + patch_size // 2)
        y2 = min(h, py + patch_size // 2)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = tissue[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, (patch_size, patch_size))
        patches.append(patch)
        valid_idx.append(i)

    assert len(patches) > 0, f"No valid patches for {sample['sample_id']}"
    msi_mat = msi_mat_full[valid_idx]
    assert msi_mat.shape[0] == len(patches)

    return {
        "msi_raw": msi_mat,
        "patches": np.array(patches),
        "n_spots": len(patches),
    }

# 4. SINGLE-SLIDE PIPELINE (train/test WITHIN that slide)
def run_single_slide_pipeline(sample, test_split=TEST_SPLIT, n_pca=N_PCA):
    print(f"\n{'='*60}")
    print(f"Processing: {sample['sample_id']}")
    print(f"{'='*60}")

    data = extract_sample_data(sample)
    n_spots = data["n_spots"]
    print(f"Extracted {n_spots} spots")

    # train/test split within this slide
    np.random.seed(RANDOM_SEED)
    test_size = max(5, int(n_spots * test_split))
    test_idx = np.random.choice(n_spots, test_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n_spots), test_idx)

    print(f"Train spots: {len(train_idx)} | Test spots: {len(test_idx)}")

    msi_train = data["msi_raw"][train_idx]
    msi_test = data["msi_raw"][test_idx]
    patches_train = data["patches"][train_idx]
    patches_test = data["patches"][test_idx]

    # Scale + PCA (train only)
    scaler_m = StandardScaler().fit(msi_train)
    msi_train_scaled = scaler_m.transform(msi_train)
    msi_test_scaled = scaler_m.transform(msi_test)

    pca = PCA(n_components=min(n_pca, msi_train_scaled.shape[1])).fit(msi_train_scaled)
    msi_pca_train = pca.transform(msi_train_scaled)
    msi_pca_test = pca.transform(msi_test_scaled)
    var_explained = np.sum(pca.explained_variance_ratio_)
    print(f"PCA: {var_explained:.1%} variance explained")

    # ResNet features
    feats_resnet_train = extract_features_batched(patches_train, resnet50, resnet_transform)
    feats_resnet_test = extract_features_batched(patches_test, resnet50, resnet_transform)
    scaler_r = StandardScaler().fit(feats_resnet_train)
    feats_resnet_train = scaler_r.transform(feats_resnet_train)
    feats_resnet_test = scaler_r.transform(feats_resnet_test)

    # UNI features
    feats_uni_train = extract_features_batched(patches_train, uni, uni_transform)
    feats_uni_test = extract_features_batched(patches_test, uni, uni_transform)
    scaler_u = StandardScaler().fit(feats_uni_train)
    feats_uni_train = scaler_u.transform(feats_uni_train)
    feats_uni_test = scaler_u.transform(feats_uni_test)

    # Ridge regression
    def train_ridge(X_train, X_test, y_train, y_test):
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return y_pred, {
            "rmse": rmse,
            "r2": r2_score(y_test, y_pred),
        }

    y_pred_resnet, metrics_resnet = train_ridge(
        feats_resnet_train, feats_resnet_test, msi_pca_train, msi_pca_test
    )
    y_pred_uni, metrics_uni = train_ridge(
        feats_uni_train, feats_uni_test, msi_pca_train, msi_pca_test
    )

    print(f"ResNet50: RMSE={metrics_resnet['rmse']:.3f}, R²={metrics_resnet['r2']:.3f}")
    print(f"UNI:      RMSE={metrics_uni['rmse']:.3f}, R²={metrics_uni['r2']:.3f}")

    # Visualization: PCA variance
    plt.figure(figsize=(10, 4))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.subplot(1, 2, 1)
    plt.plot(cum_var, marker="o", linewidth=1)
    plt.xlabel("PCA components")
    plt.ylabel("Cumulative variance")
    plt.title(f"{sample['sample_id']}: PCA variance")
    plt.grid(True)

    # Visualization: Predictions vs truth (PC0)
    k = 0
    y_true = msi_pca_test[:, k]
    plt.subplot(1, 2, 2)
    # FIX: Slice y_pred to get the k-th component
    plt.scatter(y_true, y_pred_uni[:, k], s=10, alpha=0.5, label="UNI")
    plt.scatter(y_true, y_pred_resnet[:, k], s=10, alpha=0.5, label="ResNet50")
    plt.xlabel(f"True PC{k}")
    plt.ylabel(f"Predicted PC{k}")
    plt.title(f"{sample['sample_id']}: PC0 prediction")
    plt.axline((0, 0), slope=1, color="red", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "sample_id": sample["sample_id"],
        "metrics": {"resnet": metrics_resnet, "uni": metrics_uni},
        "pca": pca,
    }

# 5. RUN OVER ALL SLIDES
all_results = []
for sample in all_samples:
    result = run_single_slide_pipeline(sample)
    all_results.append(result)

# 6. SUMMARY TABLE
print("\n" + "="*80)
print("SUMMARY ACROSS ALL SLIDES")
print("="*80)
summary_data = []
for r in all_results:
    summary_data.append({
        "Slide": r["sample_id"],
        "ResNet RMSE": f"{r['metrics']['resnet']['rmse']:.3f}",
        "ResNet R²": f"{r['metrics']['resnet']['r2']:.3f}",
        "UNI RMSE": f"{r['metrics']['uni']['rmse']:.3f}",
        "UNI R²": f"{r['metrics']['uni']['r2']:.3f}",
    })
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
