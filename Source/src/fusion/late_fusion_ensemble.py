"""
late_fusion_ensemble.py
========================
Cross-Modal Late Fusion: Best Text Model × Best Image Model.

Combines the two strongest individual classifiers in the project:
  • Text  : MiniLM-L12-v2 + MLP (CF condition)   — F1 0.956, AUC 0.979, FPR 0.059
  • Image : EfficientNet-B0 CF-no-adv              — F1 0.801, AUC 0.852, FPR 0.300

Two fusion strategies (all threshold-tuned on val, evaluated on test):
  1. Equal-weight     — p = 0.5 · p_text + 0.5 · p_image
  2. Learned-weight   — w* = argmax_{w} F1_val(w·p_text + (1-w)·p_image)

Metrics for Table 6 (paper):
  Accuracy, Precision, Recall, F1, AUC-ROC, Brier, FPR, FNR,
  EO-diff, Bootstrap-95%-CI(F1)

Saved outputs:
  cross_modal/results/late_fusion_results.json
  cross_modal/results/predictions/fusion_test_predictions.csv
  cross_modal/results/plots/fusion_comparison_bar.png
  cross_modal/results/plots/fusion_fairness_radar.png

Usage:
  /home/vslinux/.pyenv/versions/3.12.0/bin/python cross_modal/late_fusion_ensemble.py
  python cross_modal/late_fusion_ensemble.py --smoke-test   # fast, n_boot=20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
import joblib
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from canonical_splits import get_canonical_splits  # noqa: E402

DATA_CSV     = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k.csv"

# Text model artefacts
TEXT_MODEL_DIR   = PROJECT_ROOT / "text_models" / "enhanced_results" / "models"
EMBED_CACHE_DIR  = PROJECT_ROOT / "cross_modal" / "cache"

# Image model artefacts
IMAGE_MODEL_DIR  = PROJECT_ROOT / "image_models" / "models"

# Output paths
OUT_DIR       = PROJECT_ROOT / "cross_modal" / "results"
PRED_DIR      = OUT_DIR / "predictions"
PLOTS_DIR     = OUT_DIR / "plots"
for _d in (OUT_DIR, PRED_DIR, PLOTS_DIR, EMBED_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Add image_models to path so we can import model.py and data_prep.py
sys.path.insert(0, str(PROJECT_ROOT / "image_models"))

# ─── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
BOOTSTRAP_N   = 1500
BOOTSTRAP_CI  = 0.95
MINILM_MODEL  = "sentence-transformers/all-MiniLM-L12-v2"

# Text model baseline (from enhanced_results.json) — threshold tuned on val
TEXT_BASELINE_THRESH = 0.325

# Image model baseline (from evaluation_results.json) — threshold tuned on val
IMAGE_BASELINE_THRESH = 0.44

FUSION_CONDITIONS = ("ncf", "cf_no_adv", "cf")
CONDITION_LABELS = {
    "ncf": "nCF",
    "cf_no_adv": "CF-no-adv",
    "cf": "CF+GRL",
}


def _normalise_condition(condition: str) -> str:
    c = str(condition).strip().lower()
    alias = {
        "ncf": "ncf",
        "ncf_only": "ncf",
        "cf": "cf",
        "cf_grl": "cf",
        "cf+grl": "cf",
        "cf_no_adv": "cf_no_adv",
        "cf-no-adv": "cf_no_adv",
        "cfnoadv": "cf_no_adv",
    }
    if c not in alias:
        raise ValueError(f"Unsupported condition: {condition}")
    return alias[c]


def _text_model_path_for(condition: str) -> Path:
    cond = _normalise_condition(condition)
    model_name = "minilm_mlp_ncf.joblib" if cond == "ncf" else "minilm_mlp_cf.joblib"
    return TEXT_MODEL_DIR / model_name


def _image_model_path_for(condition: str) -> Path:
    cond = _normalise_condition(condition)
    return IMAGE_MODEL_DIR / f"efficientnet_{cond}.pth"


def _cond_slug(condition: str) -> str:
    return _normalise_condition(condition)

# Identity groups (from data_prep.py)
TARGET_GROUPS = [
    "race/ethnicity", "religion", "gender", "sexual_orientation",
    "national_origin/citizenship", "disability", "age", "multiple/none",
]
GROUP2ID = {g: i for i, g in enumerate(TARGET_GROUPS)}
ID2GROUP  = {i: g for i, g in enumerate(TARGET_GROUPS)}
N_GROUPS  = len(TARGET_GROUPS)

# Image directory mapping (mirrors data_prep.py)
IMAGE_DIRS = {
    "hate_race":          PROJECT_ROOT / "Hate"     / "Hate_race"     / "generated_images",
    "hate_religion":      PROJECT_ROOT / "Hate"     / "Hate_religion" / "generated_images",
    "hate_gender":        PROJECT_ROOT / "Hate"     / "Hate_Gender"   / "generated_images",
    "hate_other":         PROJECT_ROOT / "Hate"     / "Hate_Others"   / "generated_images",
    "ambiguous":          PROJECT_ROOT / "non-hate" / "generated_images-ambigious",
    "counter_speech":     PROJECT_ROOT / "non-hate" / "generated_images-counter-speech",
    "neutral_discussion": PROJECT_ROOT / "non-hate" / "generated_images-neutral",
    "offensive_non_hate": PROJECT_ROOT / "non-hate" / "generated_images-offensive-non-hate",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "figure.facecolor": "white"})


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def build_image_index() -> dict[str, str]:
    """Scan per-class image directories → {counterfactual_id: abs_path}."""
    print("  Building image index …", flush=True)
    index: dict[str, str] = {}
    for cls, img_dir in IMAGE_DIRS.items():
        if not img_dir.exists():
            print(f"  WARNING: missing {img_dir}")
            continue
        for f in img_dir.glob("*.png"):
            cf_id = f.stem          # filename without .png
            index[cf_id] = str(f)
            index[cf_id.lower()] = str(f)   # case-insensitive fallback
    print(f"  Image index: {len(index)//2:,} unique images found.")
    return index


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load 18k CSV, filter to originals, apply CANONICAL 70/15/15 stratified
    split (class_label-stratified, via canonical_splits.py).

    Critical: uses get_canonical_splits() so val/test IDs are identical
    across all pipelines (text, image, cross-modal).

    Returns (train_df, val_df, test_df) — val and test are originals only.
    """
    print("  Loading 18k dataset …", flush=True)
    df = pd.read_csv(DATA_CSV)
    origs = df[df["cf_type"] == "original"].copy()
    origs["binary_label"] = (origs["polarity"] == "hate").astype(int)
    origs["group_id"] = origs["target_group"].map(GROUP2ID).fillna(N_GROUPS - 1).astype(int)

    # Resolve image paths
    img_index = build_image_index()
    origs["image_path"] = origs["counterfactual_id"].map(img_index)
    missing = origs["image_path"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} samples have no image — they will use a blank placeholder.")
    origs["image_path"] = origs["image_path"].fillna("")

    # ── Use canonical splits (class_label-stratified) ─────────────────────
    _canon = get_canonical_splits()
    train_ids = _canon["train_ids"]
    val_ids   = _canon["val_ids"]
    test_ids  = _canon["test_ids"]

    train_o = origs[origs["original_sample_id"].isin(train_ids)].copy()
    val_o   = origs[origs["original_sample_id"].isin(val_ids)].copy()
    test_o  = origs[origs["original_sample_id"].isin(test_ids)].copy()

    # Leakage guard
    assert train_ids.isdisjoint(val_ids),  "LEAKAGE: train ∩ val!"
    assert train_ids.isdisjoint(test_ids), "LEAKAGE: train ∩ test!"
    assert val_ids.isdisjoint(test_ids),   "LEAKAGE: val ∩ test!"

    print(f"  Split → train={len(train_o):,}  val={len(val_o):,}  test={len(test_o):,}")
    print(f"  Test hate ratio: {test_o['binary_label'].mean():.2%}")
    return train_o, val_o, test_o


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TEXT INFERENCE  (MiniLM-L12-v2 → MLP)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_minilm():
    """Lazy-load MiniLM sentence encoder (singleton)."""
    if not hasattr(_get_minilm, "_model"):
        from sentence_transformers import SentenceTransformer
        print(f"  Loading MiniLM ({MINILM_MODEL}) …", flush=True)
        t0 = time.time()
        _get_minilm._model = SentenceTransformer(MINILM_MODEL)
        print(f"  MiniLM loaded in {time.time()-t0:.1f}s")
    return _get_minilm._model


def _encode(texts: list[str], cache_tag: str) -> np.ndarray:
    """Encode text list with MiniLM, using on-disk cache."""
    cache_path = EMBED_CACHE_DIR / f"fusion_{cache_tag}.npy"
    if cache_path.exists():
        emb = np.load(cache_path)
        print(f"  Embeddings loaded from cache: {cache_tag}  {emb.shape}")
        return emb
    model = _get_minilm()
    print(f"  Encoding {len(texts):,} texts [{cache_tag}] …", flush=True)
    emb = model.encode(
        texts, batch_size=128, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    np.save(cache_path, emb)
    print(f"  Cached → {cache_path}  shape={emb.shape}")
    return emb


def run_text_inference(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_model_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load minilm_mlp_cf.joblib and return (val_probs, test_probs) — P(hate).

    Returns arrays of shape (N,).
    """
    print("\n[Text Branch] Loading MiniLM + MLP …", flush=True)

    if not text_model_path.exists():
        raise FileNotFoundError(
            f"Text model not found: {text_model_path}\n"
            "Run:  python text_models/enhanced_analysis.py  first."
        )

    clf = joblib.load(text_model_path)
    print(f"  Loaded: {text_model_path.name}  (type: {type(clf).__name__})")

    # Embed texts
    val_texts  = val_df["text"].fillna("").tolist()
    test_texts = test_df["text"].fillna("").tolist()

    X_val  = _encode(val_texts,  f"val_{len(val_texts)}")
    X_test = _encode(test_texts, f"test_{len(test_texts)}")

    # Predict probabilities — class 1 = hate
    val_probs  = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    val_preds  = (val_probs  >= TEXT_BASELINE_THRESH).astype(int)
    test_preds = (test_probs >= TEXT_BASELINE_THRESH).astype(int)

    y_val  = val_df["binary_label"].values
    y_test = test_df["binary_label"].values

    print(f"  Text val  | F1={f1_score(y_val,  val_preds,  zero_division=0):.4f}  "
          f"AUC={roc_auc_score(y_val,  val_probs):.4f}")
    print(f"  Text test | F1={f1_score(y_test, test_preds, zero_division=0):.4f}  "
          f"AUC={roc_auc_score(y_test, test_probs):.4f}")

    return val_probs, test_probs


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  IMAGE INFERENCE  (EfficientNet-B0 CF-no-adv)
# ═══════════════════════════════════════════════════════════════════════════════

class _ImageDataset(Dataset):
    """Minimal image dataset for inference — no augmentation."""

    def __init__(self, paths: list[str], labels: list[int]):
        from torchvision import transforms as T
        self.paths  = paths
        self.labels = labels
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def run_image_inference(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_model_path: Path,
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load efficientnet_cf_no_adv.pth and return (val_probs, test_probs).

    Returns arrays of shape (N,) — P(hate) derived from sigmoid(logit).
    """
    print("\n[Image Branch] Loading EfficientNet-B0 (CF-no-adv) …", flush=True)

    if not image_model_path.exists():
        raise FileNotFoundError(
            f"Image model not found: {image_model_path}\n"
            "Run:  python image_models/run_all.py  first."
        )

    from model import create_model   # image_models/model.py on sys.path

    ckpt = torch.load(image_model_path, map_location=device, weights_only=False)
    use_adv = ckpt.get("use_adversarial", False)
    config  = ckpt.get("config", {})
    n_grps  = config.get("n_groups", N_GROUPS)
    dropout = config.get("dropout",  0.3)

    model = create_model(
        use_adversarial=use_adv,
        n_groups=n_grps,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Checkpoint condition : {ckpt.get('condition','cf_no_adv')}")
    print(f"  Best val F1 (stored) : {ckpt.get('best_val_f1', 'n/a')}")

    def _infer(df: pd.DataFrame) -> np.ndarray:
        paths  = df["image_path"].tolist()
        labels = df["binary_label"].tolist()
        loader = DataLoader(
            _ImageDataset(paths, labels),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        probs_list: list[float] = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="  EfficientNet", leave=False):
                imgs = imgs.to(device)
                logits, _ = model(imgs)
                probs_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.array(probs_list)

    val_probs  = _infer(val_df)
    test_probs = _infer(test_df)

    y_val  = val_df["binary_label"].values
    y_test = test_df["binary_label"].values

    val_preds  = (val_probs  >= IMAGE_BASELINE_THRESH).astype(int)
    test_preds = (test_probs >= IMAGE_BASELINE_THRESH).astype(int)

    print(f"  Image val  | F1={f1_score(y_val,  val_preds,  zero_division=0):.4f}  "
          f"AUC={roc_auc_score(y_val,  val_probs):.4f}")
    print(f"  Image test | F1={f1_score(y_test, test_preds, zero_division=0):.4f}  "
          f"AUC={roc_auc_score(y_test, test_probs):.4f}")

    return val_probs, test_probs


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  FUSION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def optimise_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Grid-search threshold on a labelled set to maximise F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.15, 0.85, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 4), round(float(best_f1), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  OUT-OF-FOLD (OOF) GENERATION FOR STACKING (Leakage-Safe)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_text_oof(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_model_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold text probabilities via 5-fold CV on TRAIN only.
    Ensures leakage-free train predictions for stacking meta-learner.
    
    Returns:
        oof_probs_train : (N_train,) — OOF predictions from 5-fold CV
        val_probs : (N_val,) — predictions on val (for validation)
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("\n  Text OOF Generation (5-fold CV on train) …")
    
    if not text_model_path.exists():
        raise FileNotFoundError(f"Text model not found: {text_model_path}")
    
    # Load pre-trained MiniLM+MLP
    clf = joblib.load(text_model_path)
    
    # Encode all texts (train, val) using cached embeddings
    train_texts = train_df["text"].fillna("").tolist()
    val_texts   = val_df["text"].fillna("").tolist()
    
    X_train = _encode(train_texts, f"train_oof_{len(train_texts)}")
    X_val   = _encode(val_texts,   f"val_{len(val_texts)}")
    
    y_train = train_df["binary_label"].values
    y_val   = val_df["binary_label"].values
    
    # 5-fold OOF on TRAIN only
    oof_probs_train = np.zeros(len(train_df), dtype=float)
    fold_scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_train, y_train), 1):
        # Use pre-trained clf to score OOF folds
        oof_probs_train[vl_idx] = clf.predict_proba(X_train[vl_idx])[:, 1]
        fold_f1 = f1_score(y_train[vl_idx], (oof_probs_train[vl_idx] >= 0.5).astype(int), zero_division=0)
        fold_scores.append(fold_f1)
        print(f"    Fold {fold_idx}: F1={fold_f1:.4f}")
    
    # Get val probs (single forward pass)
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    print(f"  OOF-train mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  Val probs: min={val_probs.min():.4f}, max={val_probs.max():.4f}, mean={val_probs.mean():.4f}")
    
    return oof_probs_train, val_probs


def generate_image_oof(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_model_path: Path,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold image probabilities via 5-fold CV on TRAIN only.
    Uses pre-trained EfficientNet-B0 checkpoint, no fine-tuning during OOF.
    
    Returns:
        oof_probs_train : (N_train,) — OOF predictions from 5-fold CV
        val_probs : (N_val,) — predictions on val (for validation)
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("\n  Image OOF Generation (5-fold CV on train, no tuning) …")
    
    if not image_model_path.exists():
        raise FileNotFoundError(
            f"Image model not found: {image_model_path}\n"
            "Run: python image_models/run_all.py  first."
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint once
    from model import create_model
    ckpt = torch.load(image_model_path, map_location=device, weights_only=False)
    use_adv = ckpt.get("use_adversarial", False)
    config  = ckpt.get("config", {})
    n_grps  = config.get("n_groups", N_GROUPS)
    dropout = config.get("dropout", 0.3)
    
    model = create_model(use_adversarial=use_adv, n_groups=n_grps, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"    Model condition: {ckpt.get('condition', 'cf_no_adv')}")
    
    def _infer_batch(df: pd.DataFrame) -> np.ndarray:
        """Inference on a dataframe, handling missing image paths gracefully."""
        paths  = df["image_path"].tolist()
        labels = df["binary_label"].tolist()
        loader = DataLoader(
            _ImageDataset(paths, labels),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        probs_list: list[float] = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="    Inferring", leave=False):
                imgs = imgs.to(device)
                logits, _ = model(imgs)
                probs_list.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.array(probs_list)
    
    y_train = train_df["binary_label"].values
    
    # 5-fold OOF on TRAIN
    oof_probs_train = np.zeros(len(train_df), dtype=float)
    fold_scores = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(train_df, y_train), 1):
        fold_val_df = train_df.iloc[vl_idx].reset_index(drop=True)
        fold_val_probs = _infer_batch(fold_val_df)
        oof_probs_train[vl_idx] = fold_val_probs
        fold_f1 = f1_score(y_train[vl_idx], (fold_val_probs >= 0.5).astype(int), zero_division=0)
        fold_scores.append(fold_f1)
        print(f"    Fold {fold_idx}: F1={fold_f1:.4f}")
    
    # Val probs
    val_probs = _infer_batch(val_df)
    
    print(f"  OOF-train mean F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"  Val probs: min={val_probs.min():.4f}, max={val_probs.max():.4f}, mean={val_probs.mean():.4f}")
    
    return oof_probs_train, val_probs


def fusion_equal_weight(
    p_text_val: np.ndarray, p_img_val: np.ndarray, y_val: np.ndarray,
    p_text_test: np.ndarray, p_img_test: np.ndarray,
) -> tuple[np.ndarray, float, dict]:
    """p = 0.5 · p_text + 0.5 · p_image. Threshold tuned on val."""
    p_val  = 0.5 * p_text_val  + 0.5 * p_img_val
    p_test = 0.5 * p_text_test + 0.5 * p_img_test
    thresh, val_f1 = optimise_threshold(y_val, p_val)
    info = {"strategy": "equal_weight", "w_text": 0.5, "w_image": 0.5,
            "threshold": thresh, "val_f1": val_f1}
    return p_test, thresh, info


def fusion_learned_weight(
    p_text_val: np.ndarray, p_img_val: np.ndarray, y_val: np.ndarray,
    p_text_test: np.ndarray, p_img_test: np.ndarray,
) -> tuple[np.ndarray, float, dict]:
    """Grid-search w ∈ {0.10, 0.15, …, 0.90} to maximise val-F1."""
    best_f1, best_w, best_thresh = 0.0, 0.5, 0.5
    for w in np.arange(0.10, 0.91, 0.05):
        p_blend = w * p_text_val + (1 - w) * p_img_val
        t, f1 = optimise_threshold(y_val, p_blend)
        if f1 > best_f1:
            best_f1, best_w, best_thresh = f1, float(w), t

    # Apply best w to test
    p_test  = best_w * p_text_test + (1 - best_w) * p_img_test
    info = {
        "strategy": "learned_weight",
        "w_text":   round(best_w, 4),
        "w_image":  round(1 - best_w, 4),
        "threshold": best_thresh,
        "val_f1":    round(best_f1, 4),
    }
    print(f"  Learned weights: w_text={best_w:.2f}  w_image={1-best_w:.2f}  "
          f"val_F1={best_f1:.4f}  threshold={best_thresh:.4f}")
    return p_test, best_thresh, info


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Full binary-classification metric suite."""
    nh = (y_true == 0)
    h  = (y_true == 1)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 and y_prob.max() != -1 else None
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   auc,
        "brier":     float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))) if auc else None,
        "fpr":       float(y_pred[nh].sum() / max(nh.sum(), 1)),
        "fnr":       float((1 - y_pred[h]).sum() / max(h.sum(), 1)),
    }


def compute_per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Per-identity-group FPR, FNR, AUC."""
    results: dict = {}
    for gid, gname in ID2GROUP.items():
        mask = (groups == gid)
        n = int(mask.sum())
        if n == 0:
            results[gname] = {"n": 0, "fpr": None, "fnr": None, "auc": None}
            continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        nh, h = (yt == 0), (yt == 1)
        fpr   = float(yp[nh].sum() / max(nh.sum(), 1)) if nh.sum() > 0 else None
        fnr   = float((1 - yp[h]).sum() / max(h.sum(), 1)) if h.sum() > 0 else None
        try:
            auc_g = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else None
        except Exception:
            auc_g = None
        results[gname] = {
            "n": n, "n_hate": int(h.sum()), "n_non_hate": int(nh.sum()),
            "fpr": round(fpr, 4) if fpr is not None else None,
            "fnr": round(fnr, 4) if fnr is not None else None,
            "auc": round(auc_g, 4) if auc_g is not None else None,
        }
    return results


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Demographic Parity diff + Equalised Odds diff."""
    pos_rates, fprs, tprs = [], [], []
    for gid in range(N_GROUPS):
        mask = (groups == gid)
        if mask.sum() < 5:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        pos_rates.append(float(yp.mean()))
        nh, h = (yt == 0), (yt == 1)
        if nh.sum() > 0:
            fprs.append(float(yp[nh].sum() / nh.sum()))
        if h.sum() > 0:
            tprs.append(float(yp[h].sum() / h.sum()))
    dp = (max(pos_rates) - min(pos_rates)) if len(pos_rates) >= 2 else None
    eo_fpr = (max(fprs) - min(fprs)) if len(fprs) >= 2 else None
    eo_tpr = (max(tprs) - min(tprs)) if len(tprs) >= 2 else None
    eo = (eo_fpr + eo_tpr) if (eo_fpr is not None and eo_tpr is not None) else None
    return {
        "demographic_parity_diff": round(dp, 4) if dp is not None else None,
        "equalised_odds_diff":     round(eo, 4) if eo is not None else None,
        "max_fpr_across_groups":   round(max(fprs), 4) if fprs else None,
        "min_fpr_across_groups":   round(min(fprs), 4) if fprs else None,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return round(float(ece), 4)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = BOOTSTRAP_N,
) -> dict:
    """Bootstrap 95 % confidence intervals on F1, AUC, FPR."""
    rng = np.random.default_rng(RANDOM_STATE)
    f1s, aucs, fprs = [], [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx  = rng.integers(0, n, n)
        yt   = y_true[idx]
        ypr  = np.clip(y_prob[idx], 0, 1)
        ypd  = (ypr >= threshold).astype(int)
        f1s.append(f1_score(yt, ypd, zero_division=0))
        try:
            aucs.append(roc_auc_score(yt, ypr))
        except Exception:
            pass
        nh  = (yt == 0)
        fprs.append(float(ypd[nh].sum() / max(nh.sum(), 1)) if nh.sum() > 0 else 0.0)

    def _ci(arr):
        lo = round(float(np.percentile(arr, 2.5)), 4)
        hi = round(float(np.percentile(arr, 97.5)), 4)
        return [lo, hi]

    return {
        "f1_ci":  _ci(f1s),
        "auc_ci": _ci(aucs) if aucs else None,
        "fpr_ci": _ci(fprs),
    }


def _round_dict(d: dict) -> dict:
    """Recursively round all floats to 4 d.p. for clean JSON."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 4)
        elif isinstance(v, np.floating):
            out[k] = round(float(v), 4)
        elif isinstance(v, dict):
            out[k] = _round_dict(v)
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fusion_comparison_bar(model_metrics: dict, save_path: Path) -> None:
    """
    Dual-axis bar chart: F1 (blue, left) and FPR (red, right) for all models.

    model_metrics : {label: {"f1": float, "fpr": float, "auc_roc": float}}
    """
    labels = list(model_metrics.keys())
    f1s    = [model_metrics[l]["f1"]    for l in labels]
    fprs   = [model_metrics[l]["fpr"]   for l in labels]

    COLOURS = {
        "Text-Only":           "#1D4ED8",
        "Image-Only":          "#F59E0B",
        "Equal Fusion":        "#10B981",
        "Learned Fusion":      "#8B5CF6",
        "Group-Aware Fusion":  "#DC2626",
    }
    colors = [COLOURS.get(l, "#6B7280") for l in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, f1s,  w, color=colors, alpha=0.85, label="F1 Score")
    bars2 = ax2.bar(x + w / 2, fprs, w, color=colors, alpha=0.45, hatch="//", label="FPR")

    ax1.set_ylabel("F1 Score", fontsize=12, color="#1D4ED8")
    ax2.set_ylabel("False-Positive Rate", fontsize=12, color="#DC2626")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="y", colors="#1D4ED8")
    ax2.tick_params(axis="y", colors="#DC2626")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, ha="right", rotation=20, fontsize=10)

    # Value labels
    for bar, v in zip(bars1, f1s):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#1D4ED8", fontweight="bold")
    for bar, v in zip(bars2, fprs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#DC2626")

    ax1.set_title("Cross-Modal Fusion: F1 vs FPR (Table 6)", fontsize=13, fontweight="bold", pad=12)

    patch1 = mpatches.Patch(color="#555555", alpha=0.85, label="F1 Score (solid)")
    patch2 = mpatches.Patch(color="#555555", alpha=0.45, hatch="//", label="FPR (hatched)")
    ax1.legend(handles=[patch1, patch2], loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_fusion_fairness_radar(model_metrics: dict, save_path: Path) -> None:
    """
    Radar chart with 4 axes: F1, 1-FPR, 1-EO-diff, and AUC.
    Higher = better on all axes.
    """
    axes_labels = ["F1", "1 − FPR", "1 − EO-diff", "AUC"]
    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]    # close the loop

    COLOURS = {
        "Text-Only":      "#1D4ED8",
        "Image-Only":     "#F59E0B",
        "Equal Fusion":   "#10B981",
        "Learned Fusion": "#8B5CF6",
    }

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for label, m in model_metrics.items():
        eo = m.get("equalised_odds_diff") or 0.0
        vals = [
            m["f1"],
            1.0 - m["fpr"],
            1.0 - min(eo, 1.0),
            m.get("auc_roc") or 0.0,
        ]
        vals += vals[:1]
        color = COLOURS.get(label, "#6B7280")
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), axes_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Fairness Radar: Cross-Modal Fusion Models", fontsize=12,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(smoke_test: bool = False, batch_size: int = 32, condition: str = "all") -> dict:
    cond = str(condition).strip().lower()
    if cond == "all":
        aggregate: dict[str, dict] = {}
        for one_cond in FUSION_CONDITIONS:
            aggregate[one_cond] = main(
                smoke_test=smoke_test,
                batch_size=batch_size,
                condition=one_cond,
            )

        consolidated_table = []
        for one_cond in FUSION_CONDITIONS:
            for row in aggregate[one_cond]["table6"]:
                consolidated_row = dict(row)
                consolidated_row["condition"] = CONDITION_LABELS[one_cond]
                consolidated_table.append(consolidated_row)

        # Backward-compatible default block (CF-no-adv), used by downstream plots/scripts.
        default_block = aggregate["cf_no_adv"]

        output = {
            "description": "Cross-Modal Late Fusion across image conditions",
            "conditions": list(FUSION_CONDITIONS),
            "condition_labels": CONDITION_LABELS,
            "table6_consolidated": consolidated_table,
            "by_condition": aggregate,
            "default_condition": "cf_no_adv",
            "table6": default_block["table6"],
            "detailed_results": default_block["detailed_results"],
            "text_model": default_block["text_model"],
            "image_model": default_block["image_model"],
            "bootstrap_n": default_block["bootstrap_n"],
            "n_val": default_block["n_val"],
            "n_test": default_block["n_test"],
        }
        out_json = OUT_DIR / "late_fusion_results.json"
        with open(out_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Consolidated results saved -> {out_json}")
        return output

    cond = _normalise_condition(condition)
    text_model_path = _text_model_path_for(cond)
    image_model_path = _image_model_path_for(cond)
    cond_slug = _cond_slug(cond)

    global BOOTSTRAP_N
    if smoke_test:
        BOOTSTRAP_N = 20
        batch_size  = 16
        print("╔════════════════════════════════════════════════════════════╗")
        print("║  LATE FUSION ENSEMBLE  —  SMOKE TEST                      ║")
        print("╚════════════════════════════════════════════════════════════╝")
    else:
        print("╔════════════════════════════════════════════════════════════╗")
        print("║  LATE FUSION ENSEMBLE                                      ║")
        print(f"║  MiniLM-L12 + MLP  ×  EfficientNet-B0 {CONDITION_LABELS[cond]:<17} ║")
        print("╚════════════════════════════════════════════════════════════╝")

    t_start = time.time()

    # ── 1. Data ────────────────────────────────────────────────────────────
    print("\n[1/8] Loading data …")
    train_df, val_df, test_df = load_dataset()
    print(f"  Train: n={len(train_df)}, Val: n={len(val_df)}, Test: n={len(test_df)}")
    y_train = train_df["binary_label"].values
    y_val   = val_df["binary_label"].values
    y_test  = test_df["binary_label"].values
    grp_train = train_df["group_id"].values
    grp_val  = val_df["group_id"].values
    grp_test = test_df["group_id"].values

    # ── 2. Leakage-safe OOF generation on TRAIN (5-fold CV, no test leakage) ─
    print("\n[2/8] Generating OOF predictions (train: 5-fold CV, no leakage) …")
    pt_train_oof, pt_val = generate_text_oof(train_df, val_df, text_model_path=text_model_path)
    pi_train_oof, pi_val = generate_image_oof(
        train_df,
        val_df,
        image_model_path=image_model_path,
        batch_size=batch_size,
    )

    # ── 3. Test inference (locked model, no CV) ────────────────────────────
    print("\n[3/8] Test inference (text & image) …")
    pt_test = run_text_inference(val_df, test_df, text_model_path=text_model_path)[1]  # Only get test probs
    pi_test = run_image_inference(
        val_df,
        test_df,
        image_model_path=image_model_path,
        batch_size=batch_size,
    )[1]  # Only get test probs

    # Check for NaN/Inf
    for name, arr in [("pt_val", pt_val), ("pi_val", pi_val),
                      ("pt_test", pt_test), ("pi_test", pi_test)]:
        if not np.isfinite(arr).all():
            bad = (~np.isfinite(arr)).sum()
            print(f"  WARNING: {bad} non-finite values in {name} — clipping.")
    # ── 4. Clip probabilities ──────────────────────────────────────────────
    print("\n[4/8] Clipping probability ranges …")
    for name, arr in [("pt_val", pt_val), ("pi_val", pi_val),
                      ("pt_test", pt_test), ("pi_test", pi_test),
                      ("pt_train_oof", pt_train_oof), ("pi_train_oof", pi_train_oof)]:
        if not np.isfinite(arr).all():
            bad = (~np.isfinite(arr)).sum()
            print(f"  WARNING: {bad} non-finite values in {name} — clipping.")
    pt_val  = np.clip(pt_val,  0.0, 1.0)
    pi_val  = np.clip(pi_val,  0.0, 1.0)
    pt_test = np.clip(pt_test, 0.0, 1.0)
    pi_test = np.clip(pi_test, 0.0, 1.0)
    pt_train_oof = np.clip(pt_train_oof, 0.0, 1.0)
    pi_train_oof = np.clip(pi_train_oof, 0.0, 1.0)

    # ── 5. Fusion strategies ───────────────────────────────────────────────
    print("\n[5/8] Computing fusion strategies …")

    # Strategy 1: Equal weight
    p_equal_train = 0.5 * pt_train_oof + 0.5 * pi_train_oof
    p_equal_val, t_equal, info_equal = fusion_equal_weight(
        pt_val, pi_val, y_val, pt_test, pi_test)
    pred_equal = (p_equal_val >= t_equal).astype(int)

    # Strategy 2: Learned weight
    p_learned_val, t_learned, info_learned = fusion_learned_weight(
        pt_val, pi_val, y_val, pt_test, pi_test)
    w_text = float(info_learned["w_text"])
    p_learned_train = w_text * pt_train_oof + (1.0 - w_text) * pi_train_oof
    pred_learned = (p_learned_val >= t_learned).astype(int)

    # ── 6. Evaluate ────────────────────────────────────────────────────────
    print("\n[6/8] Evaluating …")

    def _eval_model(
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        groups: np.ndarray,
        thresh: float,
        extra_info: dict | None = None,
    ) -> dict:
        """Full evaluation bundle for one model."""
        metrics     = compute_binary_metrics(y_true, y_pred, y_prob)
        per_group   = compute_per_group_metrics(y_true, y_pred, y_prob, groups)
        fairness    = compute_fairness_metrics(y_true, y_pred, groups)
        ece         = compute_ece(y_true, y_prob) if y_prob.max() != -1 else None
        ci          = bootstrap_ci(y_true, y_prob, thresh, n_boot=BOOTSTRAP_N)

        print(f"  {name:25s} | F1={metrics['f1']:.4f}  AUC={metrics.get('auc_roc') or 0.0:.4f}  "
              f"FPR={metrics['fpr']:.4f}  DP={fairness.get('demographic_parity_diff') or 0.0:.4f}")

        bundle = {
            "metrics":   _round_dict(metrics),
            "fairness":  _round_dict(fairness),
            "per_group": per_group,
            "ece":       ece,
            "ci":        ci,
            "threshold": thresh,
        }
        if extra_info:
            bundle["fusion_info"] = extra_info
        return bundle

    # Standalone baselines (re-evaluate on test with tuned thresholds)
    t_text,  _ = optimise_threshold(y_val, pt_val)
    t_image, _ = optimise_threshold(y_val, pi_val)

    results: dict = {}

    results["text_only"] = _eval_model(
        "Text-Only (MiniLM+MLP)",
        y_test, (pt_test >= t_text).astype(int), pt_test, grp_test, t_text,
    )
    results["image_only"] = _eval_model(
        "Image-Only (EfficientNet)",
        y_test, (pi_test >= t_image).astype(int), pi_test, grp_test, t_image,
    )
    results["equal_fusion"] = _eval_model(
        "Equal Fusion",
        y_test, pred_equal, p_equal_val, grp_test, t_equal, info_equal,
    )
    results["learned_fusion"] = _eval_model(
        "Learned Fusion",
        y_test, pred_learned, p_learned_val, grp_test, t_learned, info_learned,
    )

    # ── 7. Plots & save ────────────────────────────────────────────────────
    print("\n[7/8] Saving results …")

    # Build plot-friendly metrics dict
    plot_metrics: dict = {}
    plot_key_map = {
        "text_only":      "Text-Only",
        "image_only":     "Image-Only",
        "equal_fusion":   "Equal Fusion",
        "learned_fusion": "Learned Fusion",
    }
    for key, nice_name in plot_key_map.items():
        m = results[key]["metrics"]
        f = results[key]["fairness"]
        plot_metrics[nice_name] = {
            "f1":                      m["f1"],
            "fpr":                     m["fpr"],
            "auc_roc":                 m.get("auc_roc") or 0.0,
            "demographic_parity_diff": f.get("demographic_parity_diff") or 0.0,
            "equalised_odds_diff":     f.get("equalised_odds_diff") or 0.0,
        }

    plot_fusion_comparison_bar(plot_metrics,
                               PLOTS_DIR / "fusion_comparison_bar.png")
    plot_fusion_fairness_radar(plot_metrics,
                               PLOTS_DIR / "fusion_fairness_radar.png")

    # Table 6 summary (paper-ready)
    table6 = []
    for key, nice_name in plot_key_map.items():
        r = results[key]
        m = r["metrics"]
        ci = r["ci"]
        table6.append({
            "model":         nice_name,
            "f1":            m["f1"],
            "f1_95ci":       ci["f1_ci"],
            "auc_roc":       m.get("auc_roc"),
            "fpr":           m["fpr"],
            "fnr":           m["fnr"],
            "precision":     m["precision"],
            "recall":        m["recall"],
            "dp_diff":       r["fairness"].get("demographic_parity_diff"),
            "eo_diff":       r["fairness"].get("equalised_odds_diff"),
            "ece":           r.get("ece"),
        })

    full_output = {
        "description":       "Cross-Modal Late Fusion: MiniLM+MLP × EfficientNet",
        "condition":         cond,
        "condition_label":   CONDITION_LABELS[cond],
        "n_val":             int(len(val_df)),
        "n_test":            int(len(test_df)),
        "text_model":        str(text_model_path.name),
        "image_model":       str(image_model_path.name),
        "bootstrap_n":       BOOTSTRAP_N,
        "table6":            table6,
        "detailed_results":  results,
        "runtime_seconds":   round(time.time() - t_start, 1),
    }

    out_json = OUT_DIR / f"late_fusion_results_{cond_slug}.json"
    with open(out_json, "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"  Results saved → {out_json}")

    # ── TRAIN OOF Predictions CSV (for stacking meta-learner) ───────────────
    print("\n[8/8] Saving OOF predictions for stacking ensemble …")
    oof_csv = PRED_DIR / f"fusion_train_oof_predictions_{cond_slug}.csv"
    oof_df = train_df[["original_sample_id", "counterfactual_id",
                       "text", "class_label", "target_group",
                       "polarity", "cf_type"]].copy()
    oof_df["y_true"]           = y_train
    oof_df["p_text"]           = np.round(pt_train_oof, 6)
    oof_df["p_image"]          = np.round(pi_train_oof, 6)
    oof_df["p_equal_fusion"]   = np.round(p_equal_train, 6)
    oof_df["p_learned_fusion"] = np.round(p_learned_train, 6)
    oof_df["target_group_id"]  = grp_train
    oof_df.to_csv(oof_csv, index=False)
    print(f"  Train OOF saved → {oof_csv}  ({len(oof_df)} samples)")

    # ── TEST Predictions CSV ───────────────────────────────────────────────
    pred_csv = PRED_DIR / f"fusion_test_predictions_{cond_slug}.csv"
    pred_df = test_df[["original_sample_id", "counterfactual_id",
                        "text", "class_label", "target_group",
                        "polarity", "cf_type"]].copy()
    pred_df["y_true"]          = y_test
    pred_df["p_text"]          = np.round(pt_test, 6)
    pred_df["p_image"]         = np.round(pi_test, 6)
    pred_df["p_equal_fusion"]  = np.round(p_equal_val, 6)
    pred_df["p_learned_fusion"]= np.round(p_learned_val, 6)
    pred_df["pred_learned"]    = pred_learned
    pred_df["target_group_id"] = grp_test
    pred_df.to_csv(pred_csv, index=False)
    print(f"  Test predictions saved → {pred_csv}  ({len(pred_df)} samples)")

    # ── Summary banner ────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  TABLE 6 SUMMARY (test set, n=900)")
    print("═" * 68)
    print(f"  {'Model':<25}  {'F1':>7}  {'AUC':>7}  {'FPR':>7}  {'DP-diff':>8}")
    print("  " + "-" * 64)
    for row in table6:
        auc_str = f"{row['auc_roc']:.4f}" if row['auc_roc'] else "  —   "
        dp_str  = f"{row['dp_diff']:.4f}" if row['dp_diff'] else "  —   "
        print(f"  {row['model']:<25}  {row['f1']:>7.4f}  {auc_str:>7}  "
              f"{row['fpr']:>7.4f}  {dp_str:>8}")
    print("═" * 68)
    print(f"  Total runtime: {time.time()-t_start:.1f}s")

    return full_output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-Modal Late Fusion Ensemble")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Fast run: n_boot=20, batch_size=16")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["all", "ncf", "cf_no_adv", "cf"],
        help="Fusion condition to run. Default runs all and writes consolidated output.",
    )
    args = ap.parse_args()
    main(smoke_test=args.smoke_test, batch_size=args.batch_size, condition=args.condition)
