#!/usr/bin/env python3
"""
multi_seed_experiment.py — Multi-seed variance estimation for neural models
==========================================================================
Runs 3-seed experiments for ACM MM 2026 statistical rigor requirements.

Seeds: [42, 123, 456]
Models: MiniLM+MLP (text), EfficientNet-B0 (image x3 conditions), CrossAttn Fusion

Usage:
  python3 scripts/multi_seed_experiment.py                    # Full run
  python3 scripts/multi_seed_experiment.py --text-only        # Text MLP only
  python3 scripts/multi_seed_experiment.py --image-only       # Image only
  python3 scripts/multi_seed_experiment.py --fusion-only      # Fusion only
  python3 scripts/multi_seed_experiment.py --smoke-test       # Quick validation
  python3 scripts/multi_seed_experiment.py --seeds 42 123 456 # Custom seeds
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# ─── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from canonical_splits import get_canonical_splits  # noqa: E402

# ─── Constants ───────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [42, 123, 456]
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "multi_seed_results.json"

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
EMBED_CACHE = PROJECT_ROOT / "text_models" / "enhanced_results" / "embeddings"

# MLP architecture matching text_models/enhanced_analysis.py:
#   MLPClassifier(hidden_layer_sizes=(256, 64), activation="relu",
#                 solver="adam", alpha=1e-3, max_iter=400,
#                 early_stopping=True, validation_fraction=0.1)
MLP_HIDDEN = (256, 64)
MLP_ALPHA = 1e-3
MLP_MAX_ITER = 400

IMAGE_CONDITIONS = ["ncf", "cf_no_adv", "cf"]


# ═════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"  Device: CUDA ({name})")
        return dev
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Device: MPS (Apple Silicon)")
        return torch.device("mps")
    print("  Device: CPU")
    return torch.device("cpu")


def set_all_seeds(seed: int):
    """Deterministically seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def fmt_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def fmt_metric(values: list[float]) -> str:
    """Format mean ± std for display."""
    arr = np.array(values)
    return f"{arr.mean():.4f} ± {arr.std():.4f}"


def bootstrap_ci(values: list[float], n_boot: int = 2000,
                 ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 95% CI on the mean of a small sample."""
    arr = np.array(values)
    if len(arr) < 2:
        return (arr[0], arr[0]) if len(arr) == 1 else (np.nan, np.nan)
    rng = np.random.RandomState(42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return (lo, hi)


class ProgressTracker:
    """Track experiment progress with ETA estimation."""

    def __init__(self, total_tasks: int):
        self.total = total_tasks
        self.completed = 0
        self.start_time = time.time()
        self.task_times: list[float] = []

    def start_task(self, name: str):
        self._task_start = time.time()
        elapsed = time.time() - self.start_time
        eta = self._estimate_remaining()
        print(f"\n{'─' * 60}")
        print(f"  [{self.completed + 1}/{self.total}] {name}")
        print(f"  Elapsed: {fmt_time(elapsed)} | ETA: {eta}")
        print(f"{'─' * 60}")

    def finish_task(self):
        dt = time.time() - self._task_start
        self.task_times.append(dt)
        self.completed += 1
        print(f"  ✓ Completed in {fmt_time(dt)}")

    def _estimate_remaining(self) -> str:
        if not self.task_times:
            return "estimating…"
        avg = np.mean(self.task_times)
        remaining = (self.total - self.completed) * avg
        return fmt_time(remaining)


# ═════════════════════════════════════════════════════════════════════════════
#  TEXT MLP MULTI-SEED
# ═════════════════════════════════════════════════════════════════════════════

def _load_text_data(smoke_test: bool = False) -> dict:
    """
    Load text data + MiniLM embeddings for both nCF and CF conditions.

    Uses cached embeddings from text_models/enhanced_results/embeddings/
    when available, otherwise computes fresh.
    """
    from sentence_transformers import SentenceTransformer
    import pandas as pd

    print("\n  Loading text data (canonical splits)…")

    # ── Load CSVs ────────────────────────────────────────────────────────
    DATA_6K = PROJECT_ROOT / "src" / "counterfactual_gen" / "hate_speech_dataset_6k.csv"
    DATA_18K = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k.csv"

    splits = get_canonical_splits()
    train_ids, val_ids, test_ids = splits["train_ids"], splits["val_ids"], splits["test_ids"]

    # nCF: 6k originals
    df_6k = pd.read_csv(DATA_6K)
    df_6k["binary_label"] = (df_6k["polarity"].str.strip().str.lower() == "hate").astype(int)
    mask_ne = df_6k["text"].fillna("").apply(
        lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1) <= 0.05)
    df_6k = df_6k[mask_ne].copy()

    train_ncf = df_6k[df_6k["sample_id"].isin(train_ids)].copy()
    val_df = df_6k[df_6k["sample_id"].isin(val_ids)].copy()
    test_df = df_6k[df_6k["sample_id"].isin(test_ids)].copy()

    # CF: 18k (train uses all CF variants, val/test same originals)
    df_18k = pd.read_csv(DATA_18K)
    df_18k["binary_label"] = (df_18k["polarity"].str.strip().str.lower() == "hate").astype(int)
    mask_ne2 = df_18k["text"].fillna("").apply(
        lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1) <= 0.05)
    df_18k = df_18k[mask_ne2].copy()

    train_cf = df_18k[df_18k["original_sample_id"].isin(train_ids)].copy()

    if smoke_test:
        train_ncf = train_ncf.head(100)
        train_cf = train_cf.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # ── Encode embeddings ────────────────────────────────────────────────
    print(f"  Loading MiniLM ({MINILM_MODEL_NAME})…")
    encoder = SentenceTransformer(MINILM_MODEL_NAME)

    def _encode(texts, tag):
        cache_path = EMBED_CACHE / f"minilm_{tag}.npy"
        if cache_path.exists() and not smoke_test:
            emb = np.load(cache_path)
            if emb.shape[0] == len(texts):
                print(f"    Cached: {tag} ({emb.shape})")
                return emb
        print(f"    Encoding {len(texts):,} texts [{tag}]…")
        emb = encoder.encode(
            texts, batch_size=128, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        if not smoke_test:
            EMBED_CACHE.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, emb)
        return emb

    n_val, n_test = len(val_df), len(test_df)
    X_val = _encode(val_df["text"].fillna("").tolist(), f"val_{n_val}")
    X_test = _encode(test_df["text"].fillna("").tolist(), f"test_{n_test}")

    n_ncf = len(train_ncf)
    X_train_ncf = _encode(train_ncf["text"].fillna("").tolist(),
                          f"ncf_train_{n_ncf}")
    n_cf = len(train_cf)
    X_train_cf = _encode(train_cf["text"].fillna("").tolist(),
                         f"cf_train_{n_cf}")

    return {
        "ncf": {
            "X_train": X_train_ncf, "y_train": train_ncf["binary_label"].values,
            "X_val": X_val, "y_val": val_df["binary_label"].values,
            "X_test": X_test, "y_test": test_df["binary_label"].values,
        },
        "cf": {
            "X_train": X_train_cf, "y_train": train_cf["binary_label"].values,
            "X_val": X_val, "y_val": val_df["binary_label"].values,
            "X_test": X_test, "y_test": test_df["binary_label"].values,
        },
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray) -> dict[str, float]:
    """Compute F1, AUC, FPR, FNR from predictions."""
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {"f1": float(f1), "auc": float(auc),
            "fpr": float(fpr), "fnr": float(fnr)}


def run_text_mlp_seeds(seeds: list[int],
                       smoke_test: bool = False) -> dict[str, Any]:
    """
    Train MiniLM + MLP for each seed on both nCF and CF conditions.

    Uses sklearn.neural_network.MLPClassifier matching the architecture
    in text_models/enhanced_analysis.py: (256, 64), relu, adam, α=1e-3.
    """
    from sklearn.neural_network import MLPClassifier

    print("\n" + "=" * 60)
    print("  TEXT MLP — Multi-Seed Experiment")
    print("=" * 60)

    data = _load_text_data(smoke_test=smoke_test)
    results: dict[str, list[dict]] = {"MiniLM+MLP (nCF)": [], "MiniLM+MLP (CF)": []}
    max_iter = 2 if smoke_test else MLP_MAX_ITER

    for cond_key, cond_label in [("ncf", "MiniLM+MLP (nCF)"),
                                 ("cf", "MiniLM+MLP (CF)")]:
        d = data[cond_key]
        for seed in seeds:
            t0 = time.time()
            set_all_seeds(seed)
            print(f"\n  [{cond_label}] seed={seed} — training…", flush=True)

            clf = MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN,
                activation="relu",
                solver="adam",
                alpha=MLP_ALPHA,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            )
            clf.fit(d["X_train"], d["y_train"])

            y_prob = clf.predict_proba(d["X_test"])[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = _compute_metrics(d["y_test"], y_pred, y_prob)
            metrics["seed"] = seed
            metrics["time"] = time.time() - t0

            results[cond_label].append(metrics)
            print(f"    F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
                  f"FPR={metrics['fpr']:.4f}  FNR={metrics['fnr']:.4f}  "
                  f"({fmt_time(metrics['time'])})")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  IMAGE EFFICIENTNET MULTI-SEED
# ═════════════════════════════════════════════════════════════════════════════

def run_image_seeds(seeds: list[int],
                    smoke_test: bool = False) -> dict[str, Any]:
    """
    Train EfficientNet-B0 for each seed × condition.

    Imports from image_models package:
      - create_model, create_dataloaders, load_and_prepare, get_condition_data
      - train_model, DEFAULT_CONFIG
    """
    import torch
    from image_models.model import create_model
    from image_models.data_prep import (
        load_and_prepare, get_condition_data, create_dataloaders,
    )
    from image_models.train import train_model as train_image_model, DEFAULT_CONFIG

    print("\n" + "=" * 60)
    print("  IMAGE EFFICIENTNET-B0 — Multi-Seed Experiment")
    print("=" * 60)

    device = get_device()

    # Load and prepare data once (canonical splits, image paths)
    prepared = load_and_prepare()

    results: dict[str, list[dict]] = {}
    conditions = IMAGE_CONDITIONS

    for cond in conditions:
        label = f"EfficientNet ({cond})"
        results[label] = []

        for seed in seeds:
            t0 = time.time()
            set_all_seeds(seed)
            print(f"\n  [{label}] seed={seed}", flush=True)

            # Override config for this seed
            config = DEFAULT_CONFIG.copy()
            if smoke_test:
                config["epochs"] = 2
                config["patience"] = 2

            # Build condition data and loaders
            cond_data = get_condition_data(prepared, cond)

            if smoke_test:
                # Subsample for quick validation
                for split_key in ("train", "val", "test"):
                    cond_data[split_key] = cond_data[split_key].head(100)

            loaders = create_dataloaders(
                cond_data,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
            )

            # Train from scratch (train_model creates a fresh model internally)
            result = train_image_model(
                condition=cond,
                loaders=loaders,
                config=config,
                device=str(device),
            )

            test_m = result["test_metrics"]
            # Compute full metrics from stored predictions
            y_true = test_m["labels"]
            y_pred = test_m["predictions"]
            y_prob = test_m["probabilities"]
            metrics = _compute_metrics(y_true, y_pred, y_prob)
            metrics["seed"] = seed
            metrics["time"] = time.time() - t0

            results[label].append(metrics)
            print(f"    F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
                  f"FPR={metrics['fpr']:.4f}  FNR={metrics['fnr']:.4f}  "
                  f"({fmt_time(metrics['time'])})")

            # Free GPU memory
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  CROSS-ATTENTION FUSION MULTI-SEED (if available)
# ═════════════════════════════════════════════════════════════════════════════

def run_fusion_seeds(seeds: list[int],
                     smoke_test: bool = False) -> dict[str, Any]:
    """
    Run multi-seed experiments on cross-modal fusion if a trainable
    fusion model is available. Falls back to late-fusion (fixed-weight)
    variance estimation if no trainable fusion module exists.
    """
    print("\n" + "=" * 60)
    print("  CROSS-MODAL FUSION — Multi-Seed Experiment")
    print("=" * 60)

    results: dict[str, list[dict]] = {}

    # ── Try trainable cross-attention fusion ─────────────────────────────
    fusion_pred_csv = (PROJECT_ROOT / "cross_modal" / "results" /
                       "predictions" / "fusion_test_predictions.csv")

    if not fusion_pred_csv.exists():
        print("  ⚠ No fusion predictions found — skipping fusion experiment.")
        print(f"    Expected: {fusion_pred_csv}")
        return results

    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    print(f"  Loading fusion predictions from {fusion_pred_csv.name}")
    df = pd.read_csv(fusion_pred_csv)

    # Required columns: y_true, p_text, p_image
    required = {"y_true", "p_text", "p_image"}
    if not required.issubset(set(df.columns)):
        print(f"  ⚠ Missing columns in fusion CSV: {required - set(df.columns)}")
        return results

    y_true = df["y_true"].values
    p_text = df["p_text"].values
    p_image = df["p_image"].values
    X_fusion = np.column_stack([p_text, p_image])

    # Stacking ensemble with different seeds
    label = "Stacking Fusion"
    results[label] = []

    for seed in seeds:
        t0 = time.time()
        set_all_seeds(seed)
        print(f"\n  [{label}] seed={seed}", flush=True)

        # 5-fold CV stacking: train LR on text+image probs
        from sklearn.model_selection import StratifiedKFold

        n_folds = 3 if smoke_test else 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_probs = np.zeros(len(y_true))

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_fusion, y_true)):
            meta_clf = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
            meta_clf.fit(X_fusion[tr_idx], y_true[tr_idx])
            oof_probs[te_idx] = meta_clf.predict_proba(X_fusion[te_idx])[:, 1]

        y_pred = (oof_probs >= 0.5).astype(int)
        metrics = _compute_metrics(y_true, y_pred, oof_probs)
        metrics["seed"] = seed
        metrics["time"] = time.time() - t0

        results[label].append(metrics)
        print(f"    F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
              f"FPR={metrics['fpr']:.4f}  FNR={metrics['fnr']:.4f}  "
              f"({fmt_time(metrics['time'])})")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  RESULTS AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_results(all_results: dict[str, list[dict]],
                      seeds: list[int]) -> dict[str, Any]:
    """
    Compute per-model mean ± std and bootstrap 95% CI.

    Returns a structured dict ready for JSON serialisation.
    """
    summary: dict[str, Any] = {
        "metadata": {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "timestamp": datetime.now().isoformat(),
            "description": "Multi-seed variance estimation for neural models",
        },
        "models": {},
    }

    for model_name, runs in all_results.items():
        if not runs:
            continue

        metrics_keys = ["f1", "auc", "fpr", "fnr"]
        model_summary: dict[str, Any] = {
            "n_runs": len(runs),
            "seeds": [r["seed"] for r in runs],
            "per_seed": runs,
        }

        for mk in metrics_keys:
            values = [r[mk] for r in runs if not np.isnan(r.get(mk, np.nan))]
            if values:
                arr = np.array(values)
                ci_lo, ci_hi = bootstrap_ci(values)
                model_summary[mk] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "ci_95": [ci_lo, ci_hi],
                    "values": [float(v) for v in values],
                }

        total_time = sum(r.get("time", 0) for r in runs)
        model_summary["total_time_seconds"] = round(total_time, 1)
        summary["models"][model_name] = model_summary

    return summary


def save_results(summary: dict[str, Any]):
    """Save results to JSON."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved → {RESULTS_PATH}")


# ═════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(summary: dict[str, Any]):
    """Print a formatted ASCII summary table."""
    seeds = summary["metadata"]["seeds"]
    models = summary["models"]

    if not models:
        print("\n  No results to display.")
        return

    print()
    print("═" * 80)
    print(f"  MULTI-SEED RESULTS (seeds: {', '.join(map(str, seeds))})")
    print("═" * 80)

    # Header
    header = (f"  {'Model':<25s} │ {'F1 (mean±std)':>15s} │ "
              f"{'AUC (mean±std)':>15s} │ {'FPR (mean±std)':>15s} │ "
              f"{'FNR (mean±std)':>15s}")
    print(header)
    print("  " + "─" * 76)

    for model_name, data in models.items():
        cols = []
        for mk in ["f1", "auc", "fpr", "fnr"]:
            if mk in data:
                m, s = data[mk]["mean"], data[mk]["std"]
                cols.append(f"{m:.3f} ± {s:.3f}")
            else:
                cols.append("  —  ")

        row = f"  {model_name:<25s} │ {cols[0]:>15s} │ {cols[1]:>15s} │ {cols[2]:>15s} │ {cols[3]:>15s}"
        print(row)

    print("  " + "─" * 76)

    # CI summary
    print(f"\n  Bootstrap 95% Confidence Intervals on the Mean:")
    print(f"  {'Model':<25s} │ {'F1 CI':>20s} │ {'AUC CI':>20s}")
    print("  " + "─" * 52)

    for model_name, data in models.items():
        ci_strs = []
        for mk in ["f1", "auc"]:
            if mk in data and "ci_95" in data[mk]:
                lo, hi = data[mk]["ci_95"]
                ci_strs.append(f"[{lo:.4f}, {hi:.4f}]")
            else:
                ci_strs.append("  —  ")
        row = f"  {model_name:<25s} │ {ci_strs[0]:>20s} │ {ci_strs[1]:>20s}"
        print(row)

    print("═" * 80)

    # Total time
    total_secs = sum(
        d.get("total_time_seconds", 0) for d in models.values()
    )
    print(f"\n  Total experiment time: {fmt_time(total_secs)}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed variance estimation for neural models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help=f"Seeds to use (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Run only the text MLP experiments.",
    )
    parser.add_argument(
        "--image-only", action="store_true",
        help="Run only the image EfficientNet experiments.",
    )
    parser.add_argument(
        "--fusion-only", action="store_true",
        help="Run only the cross-modal fusion experiments.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick validation: 1 seed, 2 epochs, 100 samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds

    if args.smoke_test:
        seeds = seeds[:1]
        print("\n  ⚡ SMOKE TEST MODE — 1 seed, reduced data, 2 epochs")

    # Determine which experiments to run
    run_text = not (args.image_only or args.fusion_only)
    run_image = not (args.text_only or args.fusion_only)
    run_fusion = not (args.text_only or args.image_only)

    # Count total tasks for progress tracking
    n_tasks = 0
    if run_text:
        n_tasks += 1  # Text MLP (both conditions internally)
    if run_image:
        n_tasks += 1  # Image (all conditions internally)
    if run_fusion:
        n_tasks += 1  # Fusion

    print()
    print("╔" + "═" * 58 + "╗")
    print("║  MULTI-SEED EXPERIMENT — Neural Model Variance Estimation ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  Seeds       : {seeds}")
    print(f"  Text MLP    : {'YES' if run_text else 'SKIP'}")
    print(f"  Image EffNet: {'YES' if run_image else 'SKIP'}")
    print(f"  Fusion      : {'YES' if run_fusion else 'SKIP'}")
    print(f"  Smoke test  : {'YES' if args.smoke_test else 'NO'}")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tracker = ProgressTracker(n_tasks)
    all_results: dict[str, list[dict]] = {}
    experiment_start = time.time()

    # ── Text MLP ─────────────────────────────────────────────────────────
    if run_text:
        tracker.start_task("MiniLM + MLP (text, nCF & CF)")
        try:
            text_results = run_text_mlp_seeds(seeds, smoke_test=args.smoke_test)
            all_results.update(text_results)
        except Exception as e:
            print(f"  ✗ Text MLP failed: {e}")
            import traceback
            traceback.print_exc()
        tracker.finish_task()

    # ── Image EfficientNet ───────────────────────────────────────────────
    if run_image:
        tracker.start_task("EfficientNet-B0 (image, 3 conditions)")
        try:
            image_results = run_image_seeds(seeds, smoke_test=args.smoke_test)
            all_results.update(image_results)
        except Exception as e:
            print(f"  ✗ Image EfficientNet failed: {e}")
            import traceback
            traceback.print_exc()
        tracker.finish_task()

    # ── Fusion ───────────────────────────────────────────────────────────
    if run_fusion:
        tracker.start_task("Cross-Modal Fusion")
        try:
            fusion_results = run_fusion_seeds(seeds, smoke_test=args.smoke_test)
            all_results.update(fusion_results)
        except Exception as e:
            print(f"  ✗ Fusion failed: {e}")
            import traceback
            traceback.print_exc()
        tracker.finish_task()

    # ── Aggregate & save ─────────────────────────────────────────────────
    total_elapsed = time.time() - experiment_start
    summary = aggregate_results(all_results, seeds)
    summary["metadata"]["total_time_seconds"] = round(total_elapsed, 1)
    summary["metadata"]["total_time_human"] = fmt_time(total_elapsed)

    save_results(summary)
    print_summary_table(summary)

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Wall time: {fmt_time(total_elapsed)}")

    return summary


if __name__ == "__main__":
    main()
