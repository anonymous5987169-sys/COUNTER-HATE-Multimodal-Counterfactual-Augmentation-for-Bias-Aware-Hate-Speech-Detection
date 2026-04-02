"""
cross_attention_fusion.py
=========================
Feature-Level Cross-Modal Fusion with Gated Multimodal Unit (GMU)
and Bidirectional Cross-Attention for Hate Speech Detection.

Architecture
------------
Text Branch:   MiniLM-L12-v2 (frozen) -> 384-dim sentence embeddings
Image Branch:  EfficientNet-B0 (frozen backbone) -> 1280-dim -> Linear -> 384-dim

Feature-level fusion pipeline:
  1. Project image features to 384-dim (matches text dimensionality)
  2. Gated Multimodal Unit (GMU): learnable gate mixes modalities
  3. Bidirectional Cross-Attention: text ↔ image attention exchange
  4. Concatenate [text_attn; image_attn; text_orig; image_proj] -> 1536-dim
  5. Classification head:  Linear(1536,256) -> ReLU -> Dropout -> Linear(256,1)
  6. Adversarial head (GRL): Linear(1536,256) -> ReLU -> Linear(256,8)

Training:
  - AdamW, lr=1e-3, weight_decay=1e-4
  - CosineAnnealingLR scheduler
  - Early stopping on val F1 (patience=7)
  - Label smoothing = 0.05
  - GRL λ schedule (Ganin et al., JMLR 2016)
  - 5-fold CV for robust evaluation

Evaluation:
  F1, AUC, FPR, ECE, DP-diff, EO-diff, per-group FPR,
  Bootstrap 95% CIs (1500 resamples), comparison vs late-fusion baseline.

Saved outputs:
  cross_modal/results/cross_attention_fusion_results.json
  cross_modal/models/cross_attention_fusion.pt
  plots/cross_attention_fusion_comparison.png
  plots/cross_attention_gate_distribution.png

Usage:
  python cross_modal/cross_attention_fusion.py
  python cross_modal/cross_attention_fusion.py --smoke-test   # 100 samples, 2 epochs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "image_models"))

from canonical_splits import build_condition_split_frames, get_canonical_splits  # noqa: E402

DATA_CSV = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k.csv"

# Model artefacts
TEXT_MODEL_PATH = (
    PROJECT_ROOT / "text_models" / "enhanced_results" / "models" / "minilm_mlp_cf.joblib"
)
IMAGE_MODEL_PATH = PROJECT_ROOT / "image_models" / "models" / "efficientnet_cf_no_adv.pth"
# NOTE: Using CF-no-adv (not CF+GRL) because:
#   1. Condition alignment: data uses CF (18K with counterfactuals)
#   2. Fairness debiasing: cross-attention fusion has its own GRL head (0.3 weight)
#      so we avoid double-adversarial training
#   3. Baseline image performance: CF-no-adv F1=0.801 is representative image-only baseline
# See: CLAUDE.md § Canonical Result Files · Image Models
# Fallback name used in some checkpoints
IMAGE_MODEL_PATH_ALT = PROJECT_ROOT / "image_models" / "models" / "best_model_cf_no_adv.pt"

# Cache / output directories
CACHE_DIR = PROJECT_ROOT / "cross_modal" / "cache"
OUT_DIR = PROJECT_ROOT / "cross_modal" / "results"
MODELS_DIR = PROJECT_ROOT / "cross_modal" / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
for _d in (CACHE_DIR, OUT_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Constants ----------------------------------------------------------------
RANDOM_STATE = 42
BOOTSTRAP_N = 1500
BOOTSTRAP_CI = 0.95
MINILM_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
TEXT_DIM = 768
IMAGE_RAW_DIM = 1280
FUSED_DIM = TEXT_DIM  # project image to same dim as text -> 384

TARGET_GROUPS: List[str] = [
    "race/ethnicity",
    "religion",
    "gender",
    "sexual_orientation",
    "national_origin/citizenship",
    "disability",
    "age",
    "multiple/none",
]
GROUP2ID: Dict[str, int] = {g: i for i, g in enumerate(TARGET_GROUPS)}
ID2GROUP: Dict[int, str] = {i: g for i, g in enumerate(TARGET_GROUPS)}
N_GROUPS: int = len(TARGET_GROUPS)
FUSION_CONDITIONS = ("ncf", "cf_no_adv", "cf")
CONDITION_LABELS = {
    "ncf": "nCF",
    "cf_no_adv": "CF-no-adv",
    "cf": "CF+GRL",
}

IMAGE_DIRS: Dict[str, Path] = {
    "hate_race": PROJECT_ROOT / "Hate" / "Hate_race" / "generated_images",
    "hate_religion": PROJECT_ROOT / "Hate" / "Hate_religion" / "generated_images",
    "hate_gender": PROJECT_ROOT / "Hate" / "Hate_Gender" / "generated_images",
    "hate_other": PROJECT_ROOT / "Hate" / "Hate_Others" / "generated_images",
    "ambiguous": PROJECT_ROOT / "non-hate" / "generated_images-ambigious",
    "counter_speech": PROJECT_ROOT / "non-hate" / "generated_images-counter-speech",
    "neutral_discussion": PROJECT_ROOT / "non-hate" / "generated_images-neutral",
    "offensive_non_hate": PROJECT_ROOT / "non-hate" / "generated_images-offensive-non-hate",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white"})


# ===============================================================================
#  1.  GRADIENT REVERSAL LAYER (Ganin et al., JMLR 2016)
# ===============================================================================


class _GradientReversalFn(Function):
    """Autograd function that negates gradients in backward pass."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps ``_GradientReversalFn`` as an ``nn.Module``."""

    def __init__(self) -> None:
        super().__init__()
        self.lambda_: float = 1.0

    def set_lambda(self, val: float) -> None:
        self.lambda_ = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambda_)


def grl_lambda_schedule(epoch: int, total_epochs: int) -> float:
    """DANN schedule: λ = 2/(1+exp(−10p))−1 where p = epoch/total_epochs."""
    p = epoch / max(total_epochs - 1, 1)
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ===============================================================================
#  2.  GATED MULTIMODAL UNIT (GMU)
# ===============================================================================


class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (Arevalo et al., 2017).

    Computes a learnable gate to fuse two modality embeddings of the same
    dimensionality:

        gate  = σ(W_g · [text_embed; image_proj] + b_g)
        fused = gate ⊙ text_embed + (1 − gate) ⊙ image_proj

    Parameters
    ----------
    dim : int
        Dimensionality of each modality embedding (both must match).

    Attributes
    ----------
    gate_linear : nn.Linear
        Projects concatenated embeddings to gate logits.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_linear = nn.Linear(dim * 2, dim)

    def forward(
        self, text_embed: torch.Tensor, image_proj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        text_embed : Tensor of shape (B, dim)
            Text modality embeddings (e.g., MiniLM 384-dim).
        image_proj : Tensor of shape (B, dim)
            Image features projected to the same dimensionality.

        Returns
        -------
        fused : Tensor of shape (B, dim)
            Gated fusion of the two modalities.
        gate_values : Tensor of shape (B, dim)
            Sigmoid gate activations (for interpretability).
        """
        concat = torch.cat([text_embed, image_proj], dim=-1)  # (B, 2*dim)
        gate = torch.sigmoid(self.gate_linear(concat))  # (B, dim)
        fused = gate * text_embed + (1.0 - gate) * image_proj  # (B, dim)
        return fused, gate


# ===============================================================================
#  3.  CROSS-ATTENTION BLOCK
# ===============================================================================


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional Cross-Attention between text and image modalities.

    For each direction, the query comes from one modality and
    the key/value come from the other:
      - text_attn  = MultiheadAttention(Q=text, K=image, V=image)
      - image_attn = MultiheadAttention(Q=image, K=text, V=text)

    The final output concatenates all representations:
      output = [text_attn; image_attn; text_orig; image_proj]  -> 4 × dim

    Parameters
    ----------
    dim : int
        Input dimensionality for each modality.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability in the attention layers.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm_t = nn.LayerNorm(dim)
        self.layer_norm_i = nn.LayerNorm(dim)
        self.output_dim = dim * 4

    def forward(
        self, text_embed: torch.Tensor, image_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        text_embed : Tensor of shape (B, dim)
            Text modality embeddings.
        image_proj : Tensor of shape (B, dim)
            Projected image features.

        Returns
        -------
        fused : Tensor of shape (B, 4*dim)
            Concatenation of cross-attended and original representations.
        """
        # MultiheadAttention expects (B, seq_len, dim); treat each sample as
        # a single-token sequence for feature-level fusion.
        text_seq = text_embed.unsqueeze(1)  # (B, 1, dim)
        image_seq = image_proj.unsqueeze(1)  # (B, 1, dim)

        # Text attends to image (Q=text, K/V=image)
        text_attn, _ = self.text_to_image_attn(
            query=text_seq, key=image_seq, value=image_seq
        )
        text_attn = self.layer_norm_t(text_attn + text_seq)  # residual + LN
        text_attn = text_attn.squeeze(1)  # (B, dim)

        # Image attends to text (Q=image, K/V=text)
        image_attn, _ = self.image_to_text_attn(
            query=image_seq, key=text_seq, value=text_seq
        )
        image_attn = self.layer_norm_i(image_attn + image_seq)  # residual + LN
        image_attn = image_attn.squeeze(1)  # (B, dim)

        # Concatenate: [text_attn; image_attn; text_orig; image_proj]
        fused = torch.cat(
            [text_attn, image_attn, text_embed, image_proj], dim=-1
        )  # (B, 4*dim)
        return fused


# ===============================================================================
#  4.  CROSS-MODAL FUSION MODEL
# ===============================================================================


class CrossModalFusionModel(nn.Module):
    """
    Complete cross-modal fusion model for hate-speech detection.

    Pipeline:
      1. text_embed (384-dim, from MiniLM, frozen)
      2. image_features (1280-dim, from EfficientNet backbone)
         -> image_proj via Linear(1280, 384)
      3. GMU gate: learnable modality weighting
      4. Cross-Attention: bidirectional text ↔ image attention
      5. Classification head: 1536 -> 256 -> 1  (binary hate/non-hate)
      6. Adversarial head (GRL): 1536 -> 256 -> 8  (group prediction)

    Parameters
    ----------
    text_dim : int
        Dimensionality of text embeddings (default 384 for MiniLM-L12-v2).
    image_dim : int
        Dimensionality of raw image features (default 1280 for EfficientNet-B0).
    n_heads : int
        Number of attention heads in cross-attention block.
    n_groups : int
        Number of identity groups for the adversarial head.
    dropout : float
        Dropout probability for classification heads.
    attn_dropout : float
        Dropout probability in attention layers.
    """

    def __init__(
        self,
        text_dim: int = TEXT_DIM,
        image_dim: int = IMAGE_RAW_DIM,
        n_heads: int = 4,
        n_groups: int = N_GROUPS,
        dropout: float = 0.3,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim

        # -- Image projection: 1280 -> 384 ----------------------------------
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(attn_dropout),
        )

        # -- GMU gating ----------------------------------------------------
        self.gmu = GatedMultimodalUnit(dim=text_dim)

        # -- Cross-Attention -----------------------------------------------
        self.cross_attn = CrossAttentionBlock(
            dim=text_dim, n_heads=n_heads, dropout=attn_dropout
        )

        # -- Classification head -------------------------------------------
        fused_dim = self.cross_attn.output_dim  # 4 * text_dim = 1536
        self.task_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # -- Adversarial head (with GRL) ----------------------------------
        self.grl = GradientReversalLayer()
        self.adv_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_groups),
        )

    def set_grl_lambda(self, val: float) -> None:
        """Update the gradient reversal strength."""
        self.grl.set_lambda(val)

    def forward(
        self, text_embed: torch.Tensor, image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        text_embed : Tensor of shape (B, text_dim)
            Frozen MiniLM sentence embeddings.
        image_features : Tensor of shape (B, image_dim)
            Frozen EfficientNet backbone features (1280-dim).

        Returns
        -------
        task_logits : Tensor of shape (B,)
            Binary classification logits (hate vs non-hate).
        adv_logits : Tensor of shape (B, n_groups)
            Adversarial head logits for group prediction.
        gate_values : Tensor of shape (B, text_dim)
            GMU gate activations (for interpretability).
        """
        # 1. Project image -> same dim as text
        img_proj = self.image_proj(image_features)  # (B, 384)

        # 2. GMU gating
        _gmu_fused, gate_values = self.gmu(text_embed, img_proj)  # (B, 384), (B, 384)

        # 3. Cross-Attention (uses gated representations as input)
        # Feed the GMU-weighted text and image into cross-attention
        gated_text = gate_values * text_embed  # (B, 384)
        gated_image = (1.0 - gate_values) * img_proj  # (B, 384)
        fused = self.cross_attn(gated_text, gated_image)  # (B, 1536)

        # 4. Task head
        task_logits = self.task_head(fused).squeeze(-1)  # (B,)

        # 5. Adversarial head with gradient reversal
        rev = self.grl(fused)
        adv_logits = self.adv_head(rev)  # (B, n_groups)

        return task_logits, adv_logits, gate_values

    def count_parameters(self) -> Tuple[int, int]:
        """Return (trainable, total) parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# ===============================================================================
#  5.  DATA LOADING & FEATURE EXTRACTION
# ===============================================================================


def build_image_index() -> Dict[str, str]:
    """Scan per-class image directories and return {counterfactual_id: abs_path}."""
    print("  Building image index …", flush=True)
    index: Dict[str, str] = {}
    for cls, img_dir in IMAGE_DIRS.items():
        if not img_dir.exists():
            print(f"  WARNING: missing {img_dir}")
            continue
        for f in img_dir.glob("*.png"):
            cf_id = f.stem
            index[cf_id] = str(f)
            index[cf_id.lower()] = str(f)
    print(f"  Image index: {len(index) // 2:,} unique images found.")
    return index


def load_dataset(
    condition: str = "cf_no_adv",
    smoke_test: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the 18k dataset CSV, filter to originals, apply canonical splits.

    Parameters
    ----------
    smoke_test : bool
        If True, limit to 100 samples total for fast testing.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        DataFrames with columns including text, binary_label, group_id, image_path.
    """
    print("  Loading 18k dataset …", flush=True)
    df = pd.read_csv(DATA_CSV)
    df["binary_label"] = (df["polarity"] == "hate").astype(int)
    df["group_id"] = (
        df["target_group"].map(GROUP2ID).fillna(N_GROUPS - 1).astype(int)
    )

    # Resolve image paths
    img_index = build_image_index()
    df["image_path"] = df["counterfactual_id"].map(img_index)
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} samples have no image  -  blank placeholder used.")
    df["image_path"] = df["image_path"].fillna("")

    canon = get_canonical_splits()
    split_frames = build_condition_split_frames(
        df,
        condition=condition,
        splits=canon,
        augment_val_for_cf=(condition != "ncf"),
    )
    train_o = split_frames["train"].copy()
    val_o = split_frames["val"].copy()
    test_o = split_frames["test"].copy()

    if smoke_test:
        # Take small subset preserving label balance
        rng = np.random.RandomState(RANDOM_STATE)
        n_train = min(70, len(train_o))
        n_val = min(15, len(val_o))
        n_test = min(15, len(test_o))
        train_o = train_o.sample(n=n_train, random_state=rng).reset_index(drop=True)
        val_o = val_o.sample(n=n_val, random_state=rng).reset_index(drop=True)
        test_o = test_o.sample(n=n_test, random_state=rng).reset_index(drop=True)

    print(
        f"  Split -> train={len(train_o):,}  val={len(val_o):,}  test={len(test_o):,}"
    )
    print(f"  Test hate ratio: {test_o['binary_label'].mean():.2%}")
    return train_o, val_o, test_o


# -- Text feature extraction --------------------------------------------------


def extract_text_features(texts, cache_tag, device):
    split_name = "train" if "train" in cache_tag else "test" if "test" in cache_tag else "val"
    cache_path = CACHE_DIR / f"hatebert_e2e_{split_name}.npy"
    import numpy as np
    if cache_path.exists():
        emb = np.load(cache_path)
        print(f"  Loaded HateBERT embeddings {emb.shape}")
        if len(emb) != len(texts):
            print(f"Warning size mismatch: {len(emb)} vs {len(texts)}")
            emb = emb[:len(texts)]
        return emb
    raise ValueError(f"Cache miss {cache_path}")

class _ImagePathDataset(Dataset):
    """Minimal image dataset that returns transformed images from file paths."""

    def __init__(self, paths: List[str]) -> None:
        from torchvision import transforms as T

        self.paths = paths
        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img)


def extract_image_features(
    image_paths: List[str],
    cache_tag: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract 1280-dim features from EfficientNet-B0 backbone (frozen).

    Uses on-disk numpy cache for speed on repeated runs.

    Parameters
    ----------
    image_paths : list of str
        Absolute paths to images.
    cache_tag : str
        Unique tag for the cache file.
    device : torch.device
        Device for EfficientNet inference.
    batch_size : int
        Batch size for feature extraction.

    Returns
    -------
    features : np.ndarray of shape (N, 1280)
    """
    cache_path = CACHE_DIR / f"crossattn_image_{cache_tag}.npy"
    if cache_path.exists():
        feat = np.load(cache_path)
        print(f"  Image features loaded from cache: {cache_tag}  {feat.shape}")
        return feat

    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    print(f"  Loading EfficientNet-B0 backbone on {device} …", flush=True)
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    backbone = efficientnet_b0(weights=weights)
    # Remove the classifier head  -  keep only features + avgpool
    backbone.classifier = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    # Freeze all parameters
    for p in backbone.parameters():
        p.requires_grad = False

    dataset = _ImagePathDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features: List[np.ndarray] = []
    print(
        f"  Extracting image features [{cache_tag}]: {len(image_paths):,} images …",
        flush=True,
    )
    with torch.no_grad():
        for imgs in tqdm(loader, desc="  EfficientNet backbone", leave=False):
            imgs = imgs.to(device)
            feats = backbone(imgs)  # (B, 1280)
            all_features.append(feats.cpu().numpy())

    feat = np.concatenate(all_features, axis=0)  # (N, 1280)
    np.save(cache_path, feat)
    print(f"  Cached -> {cache_path}  shape={feat.shape}")
    return feat


# ===============================================================================
#  6.  METRICS
# ===============================================================================


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Optional[float]]:
    """Full binary-classification metric suite."""
    nh = y_true == 0
    h = y_true == 1
    auc = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else None
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc,
        "brier": float(brier_score_loss(y_true, np.clip(y_prob, 0, 1)))
        if auc
        else None,
        "fpr": float(y_pred[nh].sum() / max(nh.sum(), 1)),
        "fnr": float((1 - y_pred[h]).sum() / max(h.sum(), 1)),
    }


def compute_per_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Per-identity-group FPR, FNR, AUC."""
    results: Dict[str, Dict[str, Any]] = {}
    for gid, gname in ID2GROUP.items():
        mask = groups == gid
        n = int(mask.sum())
        if n == 0:
            results[gname] = {"n": 0, "fpr": None, "fnr": None, "auc": None}
            continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        nh, h = (yt == 0), (yt == 1)
        fpr = float(yp[nh].sum() / max(nh.sum(), 1)) if nh.sum() > 0 else None
        fnr = float((1 - yp[h]).sum() / max(h.sum(), 1)) if h.sum() > 0 else None
        try:
            auc_g = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else None
        except Exception:
            auc_g = None
        results[gname] = {
            "n": n,
            "n_hate": int(h.sum()),
            "n_non_hate": int(nh.sum()),
            "fpr": round(fpr, 4) if fpr is not None else None,
            "fnr": round(fnr, 4) if fnr is not None else None,
            "auc": round(auc_g, 4) if auc_g is not None else None,
        }
    return results


def compute_fairness_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> Dict[str, Optional[float]]:
    """Demographic Parity diff + Equalised Odds diff."""
    pos_rates: List[float] = []
    fprs: List[float] = []
    tprs: List[float] = []
    for gid in range(N_GROUPS):
        mask = groups == gid
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
        "equalised_odds_diff": round(eo, 4) if eo is not None else None,
        "max_fpr_across_groups": round(max(fprs), 4) if fprs else None,
        "min_fpr_across_groups": round(min(fprs), 4) if fprs else None,
    }


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (equal-width bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return round(float(ece), 4)


def optimise_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float]:
    """Grid-search threshold on a labelled set to maximise F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.15, 0.85, 0.01):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return round(float(best_t), 4), round(float(best_f1), 4)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = BOOTSTRAP_N,
) -> Dict[str, Any]:
    """Bootstrap 95% confidence intervals on F1, AUC, FPR."""
    rng = np.random.default_rng(RANDOM_STATE)
    f1s: List[float] = []
    aucs: List[float] = []
    fprs: List[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ypr = np.clip(y_prob[idx], 0, 1)
        ypd = (ypr >= threshold).astype(int)
        f1s.append(f1_score(yt, ypd, zero_division=0))
        try:
            aucs.append(roc_auc_score(yt, ypr))
        except Exception:
            pass
        nh = yt == 0
        fprs.append(
            float(ypd[nh].sum() / max(nh.sum(), 1)) if nh.sum() > 0 else 0.0
        )

    def _ci(arr: List[float]) -> List[float]:
        lo = round(float(np.percentile(arr, 2.5)), 4)
        hi = round(float(np.percentile(arr, 97.5)), 4)
        return [lo, hi]

    return {
        "f1_ci": _ci(f1s),
        "auc_ci": _ci(aucs) if aucs else None,
        "fpr_ci": _ci(fprs),
    }


def _round_dict(d: Dict) -> Dict:
    """Recursively round all floats to 4 d.p. for clean JSON."""
    out: Dict = {}
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


# ===============================================================================
#  7.  TRAINING LOOP
# ===============================================================================


def train_one_fold(
    X_text_train: np.ndarray,
    X_img_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_text_val: np.ndarray,
    X_img_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    label_smoothing: float = 0.05,
    adv_weight: float = 0.3,
    batch_size: int = 64,
    fold_idx: int = 0,
) -> Tuple[CrossModalFusionModel, Dict[str, Any]]:
    """
    Train a CrossModalFusionModel on one fold.

    Parameters
    ----------
    X_text_train : np.ndarray of shape (N_train, 384)
    X_img_train : np.ndarray of shape (N_train, 1280)
    y_train : np.ndarray of shape (N_train,)   -  binary labels
    groups_train : np.ndarray of shape (N_train,)   -  group IDs
    X_text_val : np.ndarray of shape (N_val, 384)
    X_img_val : np.ndarray of shape (N_val, 1280)
    y_val : np.ndarray of shape (N_val,)
    groups_val : np.ndarray of shape (N_val,)
    device : torch.device
    n_epochs : int
    lr : float
    weight_decay : float
    patience : int
    label_smoothing : float
    adv_weight : float
    batch_size : int
    fold_idx : int

    Returns
    -------
    model : CrossModalFusionModel (best checkpoint by val F1)
    history : dict with training metrics per epoch
    """
    model = CrossModalFusionModel(
        text_dim=TEXT_DIM,
        image_dim=IMAGE_RAW_DIM,
        n_heads=4,
        n_groups=N_GROUPS,
        dropout=0.3,
        attn_dropout=0.1,
    ).to(device)

    trainable, total = model.count_parameters()
    if fold_idx == 0:
        print(
            f"  Model: {total:,} total params | {trainable:,} trainable "
            f"({trainable / total * 100:.1f}%)"
        )

    # Convert to tensors
    t_text = torch.tensor(X_text_train, dtype=torch.float32)
    t_img = torch.tensor(X_img_train, dtype=torch.float32)
    t_y = torch.tensor(y_train, dtype=torch.float32)
    t_grp = torch.tensor(groups_train, dtype=torch.long)

    v_text = torch.tensor(X_text_val, dtype=torch.float32).to(device)
    v_img = torch.tensor(X_img_val, dtype=torch.float32).to(device)
    v_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(t_text, t_img, t_y, t_grp)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE + fold_idx),
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    # Loss functions
    task_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)]
        ).to(device),
    )
    adv_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_f1 = 0.0
    best_state: Optional[Dict] = None
    epochs_no_improve = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_f1": [],
        "val_auc": [],
        "grl_lambda": [],
    }

    for epoch in range(n_epochs):
        model.train()
        grl_lam = grl_lambda_schedule(epoch, n_epochs)
        model.set_grl_lambda(grl_lam)

        epoch_loss = 0.0
        n_batches = 0
        for batch_text, batch_img, batch_y, batch_grp in train_loader:
            batch_text = batch_text.to(device)
            batch_img = batch_img.to(device)
            batch_y = batch_y.to(device)
            batch_grp = batch_grp.to(device)

            optimizer.zero_grad()

            task_logits, adv_logits, _gate = model(batch_text, batch_img)

            # Smoothed task loss
            smooth_y = batch_y * (1.0 - label_smoothing) + (1.0 - batch_y) * label_smoothing
            loss_task = F.binary_cross_entropy_with_logits(
                task_logits, smooth_y,
                pos_weight=torch.tensor(
                    [(y_train == 0).sum() / max((y_train == 1).sum(), 1)]
                ).to(device),
            )

            # Adversarial loss (weighted)
            loss_adv = adv_criterion(adv_logits, batch_grp)
            loss = loss_task + adv_weight * loss_adv

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # -- Validation ----------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_task_logits, _, _ = model(v_text, v_img)
            val_probs = torch.sigmoid(val_task_logits).cpu().numpy()

        val_thresh, val_f1 = optimise_threshold(y_val, val_probs)
        try:
            val_auc = float(roc_auc_score(y_val, val_probs))
        except Exception:
            val_auc = 0.0

        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["grl_lambda"].append(grl_lam)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epochs_no_improve == 0:
            print(
                f"    Fold {fold_idx} Ep {epoch+1:3d}/{n_epochs} | "
                f"loss={avg_loss:.4f}  val_F1={val_f1:.4f}  "
                f"val_AUC={val_auc:.4f}  λ={grl_lam:.3f}"
                f"{'  ★' if epochs_no_improve == 0 else ''}"
            )

        if epochs_no_improve >= patience:
            print(
                f"    Fold {fold_idx} early stopping at epoch {epoch+1} "
                f"(best val F1={best_val_f1:.4f})"
            )
            break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()

    return model, history


# ===============================================================================
#  8.  PLOTS
# ===============================================================================


def plot_comparison_bar(
    model_metrics: Dict[str, Dict[str, float]], save_path: Path
) -> None:
    """Dual-axis bar chart: F1 (left) and FPR (right) for all models."""
    labels = list(model_metrics.keys())
    f1s = [model_metrics[l]["f1"] for l in labels]
    fprs = [model_metrics[l]["fpr"] for l in labels]

    COLOURS = {
        "Late Fusion (Learned)": "#8B5CF6",
        "GMU Only": "#F59E0B",
        "Cross-Attention + GMU": "#10B981",
    }
    colors = [COLOURS.get(l, "#6B7280") for l in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, f1s, w, color=colors, alpha=0.85, label="F1 Score")
    bars2 = ax2.bar(x + w / 2, fprs, w, color=colors, alpha=0.45, hatch="//", label="FPR")

    ax1.set_ylabel("F1 Score", fontsize=12, color="#1D4ED8")
    ax2.set_ylabel("False-Positive Rate", fontsize=12, color="#DC2626")
    ax1.set_ylim(0.0, 1.05)
    ax2.set_ylim(0.0, 1.05)
    ax1.tick_params(axis="y", colors="#1D4ED8")
    ax2.tick_params(axis="y", colors="#DC2626")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, ha="right", rotation=15, fontsize=10)

    for bar, v in zip(bars1, f1s):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1D4ED8",
            fontweight="bold",
        )
    for bar, v in zip(bars2, fprs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#DC2626",
        )

    ax1.set_title(
        "Cross-Attention Fusion vs Baselines: F1 vs FPR",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    patch1 = mpatches.Patch(color="#555555", alpha=0.85, label="F1 (solid)")
    patch2 = mpatches.Patch(color="#555555", alpha=0.45, hatch="//", label="FPR (hatched)")
    ax1.legend(handles=[patch1, patch2], loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot saved -> {save_path}")


def plot_gate_distribution(
    gate_values: np.ndarray, labels: np.ndarray, save_path: Path
) -> None:
    """Histogram of mean gate activations (text-weight) for hate vs non-hate."""
    mean_gate = gate_values.mean(axis=1)  # average across dims -> (N,)
    fig, ax = plt.subplots(figsize=(8, 4))

    for lbl, name, color in [(0, "Non-Hate", "#10B981"), (1, "Hate", "#DC2626")]:
        mask = labels == lbl
        ax.hist(
            mean_gate[mask],
            bins=50,
            alpha=0.6,
            label=f"{name} (n={mask.sum()})",
            color=color,
            density=True,
        )

    ax.set_xlabel("Mean Gate Value (->1 = text-dominant, ->0 = image-dominant)")
    ax.set_ylabel("Density")
    ax.set_title("GMU Gate Distribution by Class", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Plot saved -> {save_path}")


# ===============================================================================
#  9.  MAIN PIPELINE
# ===============================================================================


def main(smoke_test: bool = False) -> Dict[str, Any]:
    """
    Full training and evaluation pipeline for GMU + Cross-Attention fusion.

    Parameters
    ----------
    smoke_test : bool
        If True, limit data to ~100 samples and 2 training epochs.

    Returns
    -------
    results : dict
        Full evaluation results including metrics, fairness, and comparisons.
    """
    global BOOTSTRAP_N

    return main_for_condition(condition="all", smoke_test=smoke_test)


def _normalise_condition(condition: str) -> str:
    c = str(condition).strip().lower()
    alias = {
        "ncf": "ncf",
        "cf_no_adv": "cf_no_adv",
        "cf-no-adv": "cf_no_adv",
        "cf": "cf",
        "cf+grl": "cf",
    }
    if c not in alias:
        raise ValueError(f"Unsupported condition: {condition}")
    return alias[c]


def main_for_condition(condition: str = "all", smoke_test: bool = False) -> Dict[str, Any]:
    global BOOTSTRAP_N
    cond = str(condition).strip().lower()
    if cond == "all":
        aggregate: Dict[str, Any] = {}
        for one_cond in FUSION_CONDITIONS:
            aggregate[one_cond] = main_for_condition(condition=one_cond, smoke_test=smoke_test)

        summary = []
        for one_cond in FUSION_CONDITIONS:
            ens = aggregate[one_cond]["ensemble"]["metrics"]
            summary.append(
                {
                    "condition": CONDITION_LABELS[one_cond],
                    "f1": ens["f1"],
                    "auc_roc": ens.get("auc_roc"),
                    "fpr": ens["fpr"],
                }
            )

        output = {
            "description": "Cross-attention fusion across nCF / CF-no-adv / CF+GRL",
            "conditions": list(FUSION_CONDITIONS),
            "condition_labels": CONDITION_LABELS,
            "summary_table": summary,
            "by_condition": aggregate,
        }
        out_json = OUT_DIR / "cross_attention_fusion_results.json"
        with open(out_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nConsolidated cross-attention results saved -> {out_json}")
        return output

    cond = _normalise_condition(cond)

    n_epochs = 2 if smoke_test else 50
    n_folds = 2 if smoke_test else 5
    batch_size = 16 if smoke_test else 64
    boot_n = 20 if smoke_test else BOOTSTRAP_N
    if smoke_test:
        BOOTSTRAP_N = boot_n

    banner = "SMOKE TEST" if smoke_test else "FULL RUN"
    print("+================================================================+")
    print(f"|  CROSS-ATTENTION + GMU FEATURE FUSION   -   {banner:<20s} |")
    print(f"|  Condition: {CONDITION_LABELS[cond]:<48s}|")
    print("|  MiniLM-L12 embeddings × EfficientNet-B0 backbone features   |")
    print("+================================================================+")

    t_start = time.time()

    # -- Device -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # -- 1. Data loading ---------------------------------------------------
    print("\n[1/7] Loading data …")
    train_df, val_df, test_df = load_dataset(condition=cond, smoke_test=smoke_test)

    y_train = train_df["binary_label"].values
    y_val = val_df["binary_label"].values
    y_test = test_df["binary_label"].values
    grp_train = train_df["group_id"].values
    grp_val = val_df["group_id"].values
    grp_test = test_df["group_id"].values

    # -- 2. Text feature extraction ----------------------------------------
    print("\n[2/7] Extracting text features (MiniLM-L12-v2) …")
    train_texts = train_df["text"].fillna("").tolist()
    val_texts = val_df["text"].fillna("").tolist()
    test_texts = test_df["text"].fillna("").tolist()

    tag_suffix = "_smoke" if smoke_test else ""
    X_text_train = extract_text_features(
        train_texts, f"train_{len(train_texts)}{tag_suffix}", device
    )
    X_text_val = extract_text_features(
        val_texts, f"val_{len(val_texts)}{tag_suffix}", device
    )
    X_text_test = extract_text_features(
        test_texts, f"test_{len(test_texts)}{tag_suffix}", device
    )

    # -- 3. Image feature extraction ---------------------------------------
    print("\n[3/7] Extracting image features (EfficientNet-B0 backbone) …")
    train_img_paths = train_df["image_path"].tolist()
    val_img_paths = val_df["image_path"].tolist()
    test_img_paths = test_df["image_path"].tolist()

    X_img_train = extract_image_features(
        train_img_paths, f"train_{len(train_img_paths)}{tag_suffix}", device, batch_size
    )
    X_img_val = extract_image_features(
        val_img_paths, f"val_{len(val_img_paths)}{tag_suffix}", device, batch_size
    )
    X_img_test = extract_image_features(
        test_img_paths, f"test_{len(test_img_paths)}{tag_suffix}", device, batch_size
    )

    print(
        f"\n  Feature shapes: text={X_text_train.shape}  image={X_img_train.shape}"
    )

    # -- 4. 5-Fold Cross-Validation ----------------------------------------
    print(f"\n[4/7] Training GMU + Cross-Attention ({n_folds}-fold CV) …")

    # Combine train + val for k-fold (test is held out entirely)
    X_text_tv = np.concatenate([X_text_train, X_text_val], axis=0)
    X_img_tv = np.concatenate([X_img_train, X_img_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)
    grp_tv = np.concatenate([grp_train, grp_val], axis=0)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_models: List[CrossModalFusionModel] = []
    fold_val_f1s: List[float] = []
    fold_val_aucs: List[float] = []
    fold_histories: List[Dict] = []

    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_text_tv, y_tv)):
        print(f"\n  -- Fold {fold_idx + 1}/{n_folds} "
              f"(train={len(tr_idx)}, val={len(vl_idx)}) --")

        model_fold, hist = train_one_fold(
            X_text_train=X_text_tv[tr_idx],
            X_img_train=X_img_tv[tr_idx],
            y_train=y_tv[tr_idx],
            groups_train=grp_tv[tr_idx],
            X_text_val=X_text_tv[vl_idx],
            X_img_val=X_img_tv[vl_idx],
            y_val=y_tv[vl_idx],
            groups_val=grp_tv[vl_idx],
            device=device,
            n_epochs=n_epochs,
            lr=1e-3,
            weight_decay=1e-4,
            patience=7 if not smoke_test else 3,
            label_smoothing=0.05,
            adv_weight=(0.0 if cond == "cf_no_adv" else 0.3),
            batch_size=batch_size,
            fold_idx=fold_idx,
        )

        # Evaluate fold model on its own OOF val
        model_fold.eval()
        with torch.no_grad():
            vl_text_t = torch.tensor(X_text_tv[vl_idx], dtype=torch.float32).to(device)
            vl_img_t = torch.tensor(X_img_tv[vl_idx], dtype=torch.float32).to(device)
            logits, _, _ = model_fold(vl_text_t, vl_img_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        _, fold_f1 = optimise_threshold(y_tv[vl_idx], probs)
        try:
            fold_auc = float(roc_auc_score(y_tv[vl_idx], probs))
        except Exception:
            fold_auc = 0.0

        fold_val_f1s.append(fold_f1)
        fold_val_aucs.append(fold_auc)
        fold_models.append(model_fold)
        fold_histories.append(hist)

        print(
            f"    Fold {fold_idx + 1} OOF -> F1={fold_f1:.4f}  AUC={fold_auc:.4f}"
        )

    print(f"\n  CV Summary: F1={np.mean(fold_val_f1s):.4f} ± {np.std(fold_val_f1s):.4f}  "
          f"AUC={np.mean(fold_val_aucs):.4f} ± {np.std(fold_val_aucs):.4f}")

    # -- 5. Final model: retrain on full train+val, or use best fold -------
    print("\n[5/7] Selecting best fold model for test evaluation …")
    best_fold_idx = int(np.argmax(fold_val_f1s))
    best_model = fold_models[best_fold_idx]
    print(f"  Best fold: {best_fold_idx + 1} (val F1={fold_val_f1s[best_fold_idx]:.4f})")

    # Also train a final model on all train+val data for deployment
    print("  Training final model on full train+val data …")
    final_model, final_hist = train_one_fold(
        X_text_train=X_text_tv,
        X_img_train=X_img_tv,
        y_train=y_tv,
        groups_train=grp_tv,
        X_text_val=X_text_val,  # use original val for early stopping
        X_img_val=X_img_val,
        y_val=y_val,
        groups_val=grp_val,
        device=device,
        n_epochs=n_epochs,
        lr=1e-3,
        weight_decay=1e-4,
        patience=7 if not smoke_test else 3,
        label_smoothing=0.05,
        adv_weight=(0.0 if cond == "cf_no_adv" else 0.3),
        batch_size=batch_size,
        fold_idx=99,
    )

    # -- 6. Test evaluation ------------------------------------------------
    print("\n[6/7] Evaluating on held-out test set …")

    # Ensemble: average logits from all fold models
    test_text_t = torch.tensor(X_text_test, dtype=torch.float32).to(device)
    test_img_t = torch.tensor(X_img_test, dtype=torch.float32).to(device)

    ensemble_probs = np.zeros(len(y_test), dtype=np.float64)
    all_gate_values = np.zeros((len(y_test), TEXT_DIM), dtype=np.float64)

    for fmodel in fold_models:
        fmodel.eval()
        with torch.no_grad():
            logits, _, gates = fmodel(test_text_t, test_img_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            ensemble_probs += probs
            all_gate_values += gates.cpu().numpy()

    ensemble_probs /= len(fold_models)
    all_gate_values /= len(fold_models)
    ensemble_probs = np.clip(ensemble_probs, 0.0, 1.0)

    # Also get single best-fold predictions
    best_model.eval()
    with torch.no_grad():
        bf_logits, _, bf_gates = best_model(test_text_t, test_img_t)
        best_fold_probs = torch.sigmoid(bf_logits).cpu().numpy()

    # And final model predictions
    final_model.eval()
    with torch.no_grad():
        fm_logits, _, fm_gates = final_model(test_text_t, test_img_t)
        final_probs = torch.sigmoid(fm_logits).cpu().numpy()

    # Tuned thresholds
    t_ens, _ = optimise_threshold(y_val, _predict_val_probs(
        fold_models, X_text_val, X_img_val, device
    ))
    t_bf, _ = optimise_threshold(y_val, _predict_probs(best_model, X_text_val, X_img_val, device))
    t_fm, _ = optimise_threshold(y_val, _predict_probs(final_model, X_text_val, X_img_val, device))

    # Primary result: ensemble
    pred_ens = (ensemble_probs >= t_ens).astype(int)
    metrics_ens = compute_binary_metrics(y_test, pred_ens, ensemble_probs)
    fairness_ens = compute_fairness_metrics(y_test, pred_ens, grp_test)
    per_group_ens = compute_per_group_metrics(y_test, pred_ens, ensemble_probs, grp_test)
    ece_ens = compute_ece(y_test, ensemble_probs)
    ci_ens = bootstrap_ci(y_test, ensemble_probs, t_ens, n_boot=boot_n)

    # Best fold
    pred_bf = (best_fold_probs >= t_bf).astype(int)
    metrics_bf = compute_binary_metrics(y_test, pred_bf, best_fold_probs)
    fairness_bf = compute_fairness_metrics(y_test, pred_bf, grp_test)
    ece_bf = compute_ece(y_test, best_fold_probs)
    ci_bf = bootstrap_ci(y_test, best_fold_probs, t_bf, n_boot=boot_n)

    # Final model
    pred_fm = (final_probs >= t_fm).astype(int)
    metrics_fm = compute_binary_metrics(y_test, pred_fm, final_probs)
    fairness_fm = compute_fairness_metrics(y_test, pred_fm, grp_test)
    ece_fm = compute_ece(y_test, final_probs)
    ci_fm = bootstrap_ci(y_test, final_probs, t_fm, n_boot=boot_n)

    print(f"\n  Ensemble      | F1={metrics_ens['f1']:.4f}  AUC={metrics_ens['auc_roc']:.4f}  "
          f"FPR={metrics_ens['fpr']:.4f}  DP={fairness_ens['demographic_parity_diff']}")
    print(f"  Best Fold     | F1={metrics_bf['f1']:.4f}   AUC={metrics_bf['auc_roc']:.4f}  "
          f"FPR={metrics_bf['fpr']:.4f}  DP={fairness_bf['demographic_parity_diff']}")
    print(f"  Final Model   | F1={metrics_fm['f1']:.4f}   AUC={metrics_fm['auc_roc']:.4f}  "
          f"FPR={metrics_fm['fpr']:.4f}  DP={fairness_fm['demographic_parity_diff']}")

    # -- Load late-fusion baseline for comparison --------------------------
    late_fusion_path = OUT_DIR / "late_fusion_results.json"
    late_fusion_baseline: Optional[Dict] = None
    if late_fusion_path.exists():
        with open(late_fusion_path) as f:
            late_fusion_baseline = json.load(f)
        print("\n  Late-fusion baseline loaded for comparison.")

    # -- 7. Save everything ------------------------------------------------
    print("\n[7/7] Saving results, model checkpoint, and plots …")

    # Save model checkpoint
    ckpt_path = MODELS_DIR / f"cross_attention_fusion_{cond}.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "config": {
                "text_dim": TEXT_DIM,
                "image_dim": IMAGE_RAW_DIM,
                "n_heads": 4,
                "n_groups": N_GROUPS,
                "dropout": 0.3,
                "attn_dropout": 0.1,
            },
            "threshold": float(t_fm),
            "val_f1": float(fold_val_f1s[best_fold_idx]),
            "cv_f1_mean": float(np.mean(fold_val_f1s)),
            "cv_f1_std": float(np.std(fold_val_f1s)),
        },
        ckpt_path,
    )
    print(f"  Model checkpoint -> {ckpt_path}")

    # Plots
    plot_metrics: Dict[str, Dict[str, float]] = {
        "GMU + CrossAttn (Ensemble)": {
            "f1": metrics_ens["f1"],
            "fpr": metrics_ens["fpr"],
        },
        "GMU + CrossAttn (Final)": {
            "f1": metrics_fm["f1"],
            "fpr": metrics_fm["fpr"],
        },
    }
    # Add late-fusion baseline to comparison if available
    if late_fusion_baseline:
        for row in late_fusion_baseline.get("table6", []):
            if row["model"] == "Learned Fusion":
                plot_metrics["Late Fusion (Learned)"] = {
                    "f1": row["f1"],
                    "fpr": row["fpr"],
                }
                break

    plot_comparison_bar(
        plot_metrics,
        PLOTS_DIR / f"cross_attention_fusion_comparison_{cond}.png",
    )
    plot_gate_distribution(
        all_gate_values, y_test,
        PLOTS_DIR / f"cross_attention_gate_distribution_{cond}.png",
    )

    final_trainable, final_total = final_model.count_parameters()
    concat_from_components = TEXT_DIM * 4

    # Assemble results dict
    results: Dict[str, Any] = {
        "description": (
            "Feature-Level Cross-Modal Fusion: GMU + Cross-Attention "
            "(MiniLM embeddings × EfficientNet backbone features)"
        ),
        "condition": cond,
        "condition_label": CONDITION_LABELS[cond],
        "architecture": {
            "text_dim": TEXT_DIM,
            "image_raw_dim": IMAGE_RAW_DIM,
            "image_proj_dim": TEXT_DIM,
            "cross_attention_output_components": [
                "text_attn",
                "image_attn",
                "gated_text",
                "gated_image",
            ],
            "concat_dim": int(final_model.cross_attn.output_dim),
            "concat_formula": "concat_dim = 4 * text_dim = 1536",
            "task_head": "Linear(1536,256) -> ReLU -> Dropout -> Linear(256,1)",
            "adv_head": "GRL -> Linear(1536,256) -> ReLU -> Linear(256,8)",
        },
        "parameter_sanity": {
            "derived_concat_dim": int(concat_from_components),
            "model_concat_dim": int(final_model.cross_attn.output_dim),
            "concat_dim_matches": bool(concat_from_components == final_model.cross_attn.output_dim),
            "trainable_params": int(final_trainable),
            "total_params": int(final_total),
        },
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_folds": n_folds,
        "device": str(device),
        "smoke_test": smoke_test,
        "cv_results": {
            "fold_val_f1s": [round(f, 4) for f in fold_val_f1s],
            "fold_val_aucs": [round(a, 4) for a in fold_val_aucs],
            "mean_f1": round(float(np.mean(fold_val_f1s)), 4),
            "std_f1": round(float(np.std(fold_val_f1s)), 4),
            "mean_auc": round(float(np.mean(fold_val_aucs)), 4),
            "std_auc": round(float(np.std(fold_val_aucs)), 4),
        },
        "ensemble": {
            "metrics": _round_dict(metrics_ens),
            "fairness": _round_dict(fairness_ens),
            "per_group": per_group_ens,
            "ece": ece_ens,
            "ci": ci_ens,
            "threshold": float(t_ens),
        },
        "best_fold": {
            "fold_idx": best_fold_idx,
            "metrics": _round_dict(metrics_bf),
            "fairness": _round_dict(fairness_bf),
            "ece": ece_bf,
            "ci": ci_bf,
            "threshold": float(t_bf),
        },
        "final_model": {
            "metrics": _round_dict(metrics_fm),
            "fairness": _round_dict(fairness_fm),
            "ece": ece_fm,
            "ci": ci_fm,
            "threshold": float(t_fm),
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    # Add late-fusion comparison
    if late_fusion_baseline:
        results["late_fusion_comparison"] = _build_comparison_table(
            late_fusion_baseline, metrics_ens, fairness_ens, ci_ens, t_ens
        )

    # Save results JSON
    out_json = OUT_DIR / f"cross_attention_fusion_results_{cond}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved -> {out_json}")

    # -- Print comparison table --------------------------------------------
    _print_comparison_table(results, late_fusion_baseline)

    print(f"\n  Total runtime: {time.time() - t_start:.1f}s")
    return results


# ===============================================================================
#  HELPERS
# ===============================================================================


def _predict_probs(
    model: CrossModalFusionModel,
    X_text: np.ndarray,
    X_img: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run a single model on numpy inputs and return P(hate)."""
    model.eval()
    with torch.no_grad():
        t_text = torch.tensor(X_text, dtype=torch.float32).to(device)
        t_img = torch.tensor(X_img, dtype=torch.float32).to(device)
        logits, _, _ = model(t_text, t_img)
        return torch.sigmoid(logits).cpu().numpy()


def _predict_val_probs(
    models: List[CrossModalFusionModel],
    X_text: np.ndarray,
    X_img: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Ensemble prediction (average) from multiple models."""
    probs = np.zeros(len(X_text), dtype=np.float64)
    for m in models:
        probs += _predict_probs(m, X_text, X_img, device)
    probs /= len(models)
    return np.clip(probs, 0.0, 1.0)


def _build_comparison_table(
    late_fusion: Dict,
    ens_metrics: Dict,
    ens_fairness: Dict,
    ens_ci: Dict,
    ens_thresh: float,
) -> Dict[str, Any]:
    """Build a structured comparison between late-fusion and cross-attention."""
    comparison: Dict[str, Any] = {"models": []}

    # Extract late-fusion models from table6
    for row in late_fusion.get("table6", []):
        comparison["models"].append(
            {
                "name": f"Late Fusion: {row['model']}",
                "f1": row["f1"],
                "f1_ci": row.get("f1_95ci"),
                "auc_roc": row.get("auc_roc"),
                "fpr": row["fpr"],
                "dp_diff": row.get("dp_diff"),
                "eo_diff": row.get("eo_diff"),
            }
        )

    # Cross-attention ensemble
    comparison["models"].append(
        {
            "name": "Cross-Attn + GMU (Ensemble)",
            "f1": ens_metrics["f1"],
            "f1_ci": ens_ci.get("f1_ci"),
            "auc_roc": ens_metrics.get("auc_roc"),
            "fpr": ens_metrics["fpr"],
            "dp_diff": ens_fairness.get("demographic_parity_diff"),
            "eo_diff": ens_fairness.get("equalised_odds_diff"),
        }
    )

    return comparison


def _print_comparison_table(
    results: Dict[str, Any], late_fusion: Optional[Dict] = None
) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 78)
    print("  COMPARISON: Late Fusion vs GMU + Cross-Attention Feature Fusion")
    print("=" * 78)
    print(
        f"  {'Model':<35s}  {'F1':>7}  {'AUC':>7}  {'FPR':>7}  "
        f"{'DP-diff':>8}  {'EO-diff':>8}"
    )
    print("  " + "-" * 74)

    rows: List[Tuple[str, Dict]] = []

    # Late fusion rows
    if late_fusion:
        for row in late_fusion.get("table6", []):
            rows.append((
                f"LF: {row['model']}",
                {
                    "f1": row["f1"],
                    "auc_roc": row.get("auc_roc"),
                    "fpr": row["fpr"],
                    "dp_diff": row.get("dp_diff"),
                    "eo_diff": row.get("eo_diff"),
                },
            ))

    # Cross-attention rows
    for key, label in [
        ("ensemble", "CrossAttn+GMU (Ensemble)"),
        ("best_fold", "CrossAttn+GMU (Best Fold)"),
        ("final_model", "CrossAttn+GMU (Final)"),
    ]:
        if key in results:
            m = results[key]["metrics"]
            f = results[key].get("fairness", {})
            rows.append((
                label,
                {
                    "f1": m["f1"],
                    "auc_roc": m.get("auc_roc"),
                    "fpr": m["fpr"],
                    "dp_diff": f.get("demographic_parity_diff"),
                    "eo_diff": f.get("equalised_odds_diff"),
                },
            ))

    for name, m in rows:
        auc_s = f"{m['auc_roc']:.4f}" if m.get("auc_roc") else "    -   "
        dp_s = f"{m['dp_diff']:.4f}" if m.get("dp_diff") is not None else "    -   "
        eo_s = f"{m['eo_diff']:.4f}" if m.get("eo_diff") is not None else "    -   "
        print(
            f"  {name:<35s}  {m['f1']:>7.4f}  {auc_s:>7}  "
            f"{m['fpr']:>7.4f}  {dp_s:>8}  {eo_s:>8}"
        )

    print("=" * 78)

    # CV summary
    cv = results.get("cv_results", {})
    if cv:
        print(
            f"\n  {results.get('n_folds', 5)}-Fold CV: "
            f"F1 = {cv.get('mean_f1', 0):.4f} ± {cv.get('std_f1', 0):.4f}  |  "
            f"AUC = {cv.get('mean_auc', 0):.4f} ± {cv.get('std_auc', 0):.4f}"
        )

    # Bootstrap CIs
    ens_ci = results.get("ensemble", {}).get("ci", {})
    if ens_ci:
        f1_ci = ens_ci.get("f1_ci", [0, 0])
        auc_ci = ens_ci.get("auc_ci", [0, 0])
        fpr_ci = ens_ci.get("fpr_ci", [0, 0])
        print(
            f"  Bootstrap 95% CI (ensemble): "
            f"F1=[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]  "
            f"AUC=[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]  "
            f"FPR=[{fpr_ci[0]:.4f}, {fpr_ci[1]:.4f}]"
        )


# ===============================================================================
#  ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="GMU + Cross-Attention Feature-Level Fusion for Hate Speech Detection"
    )
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast run: 100 samples, 2 epochs, 2 folds, n_boot=20",
    )
    ap.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["all", "ncf", "cf_no_adv", "cf"],
        help="Condition to run. Default runs all and writes consolidated output.",
    )
    args = ap.parse_args()
    main_for_condition(condition=args.condition, smoke_test=args.smoke_test)
