#!/usr/bin/env python3
"""
stacking_ensemble.py — Feature-level Stacking Ensemble: Text x Image
=====================================================================

Uses train OOF base predictions and a held-out test prediction file:
    - fusion_train_oof_predictions.csv  (train, for meta-learner fitting)
    - fusion_test_predictions.csv       (test, evaluated once)

Protocol (leakage-safe):
    1) Build OOF train diagnostics via 5-fold CV on TRAIN only.
    2) Select/tune meta-learner on TRAIN only.
    3) Fit meta-learner on full TRAIN features.
    4) Evaluate once on TEST (no CV on test set).

Level 0 (Base learners — already trained and exported as probabilities):
  - Text  : MiniLM-L12-v2 -> MLP  -> p_text
  - Image : EfficientNet-B0 CF    -> p_image

Level 1 (Meta-learner):
  9-d polynomial feature vector built from (p_text, p_image).
  Candidate meta-learners: LR, GradientBoosting, MLP(32,16), ExtraTrees

Outputs:
  cross_modal/results/stacking_ensemble_results.json
    cross_modal/results/predictions/stacking_predictions_test.csv
    cross_modal/results/predictions/stacking_train_oof_predictions.csv
  cross_modal/results/plots/stacking_comparison.png
  cross_modal/results/plots/stacking_reliability.png
  cross_modal/results/plots/stacking_per_group_fpr.png
"""

from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss,
)

# ── Paths ──────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT / "cross_modal" / "results" / "predictions"
OUT_DIR = PROJECT / "cross_modal" / "results"
PRED_DIR = OUT_DIR / "predictions"
PLOTS_DIR = OUT_DIR / "plots"
for _d in (OUT_DIR, PRED_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
BOOTSTRAP_N = 1500
FUSION_CONDITIONS = ("ncf", "cf_no_adv", "cf")
CONDITION_LABELS = {
    "ncf": "nCF",
    "cf_no_adv": "CF-no-adv",
    "cf": "CF+GRL",
}

TARGET_GROUPS = [
    "race/ethnicity", "religion", "gender", "sexual_orientation",
    "national_origin/citizenship", "disability", "age", "multiple/none",
]

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white"})


# ═══ 1. DATA LOADING ═══════════════════════════════════════════════

REQUIRED_COLS = {
    "original_sample_id", "counterfactual_id", "text", "class_label",
    "target_group", "y_true", "p_text", "p_image", "p_learned_fusion",
}


def _prediction_paths(condition: str) -> tuple[Path, Path]:
    train_csv = PREDICTIONS_DIR / f"fusion_train_oof_predictions_{condition}.csv"
    test_csv = PREDICTIONS_DIR / f"fusion_test_predictions_{condition}.csv"
    return train_csv, test_csv


def _load_split(path: Path, split_name: str, condition: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split_name} fusion prediction file: {path}\n"
            "Expected leakage-safe stacking inputs:\n"
            f"  - cross_modal/results/predictions/fusion_train_oof_predictions_{condition}.csv\n"
            f"  - cross_modal/results/predictions/fusion_test_predictions_{condition}.csv"
        )
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"{split_name} file missing required columns: {missing}")
    return df


def _extract_arrays(df: pd.DataFrame):
    pt = df["p_text"].values.astype(float)
    pi = df["p_image"].values.astype(float)
    y = df["y_true"].values.astype(int)
    grp = df["target_group"].values
    plf = df["p_learned_fusion"].values.astype(float)
    return np.clip(pt, 0, 1), np.clip(pi, 0, 1), y, grp, np.clip(plf, 0, 1)


def load_data(condition: str):
    """Load leakage-safe train/test fusion prediction files."""
    train_csv, test_csv = _prediction_paths(condition)
    df_train = _load_split(train_csv, "train", condition)
    df_test = _load_split(test_csv, "test", condition)

    overlap = set(df_train["original_sample_id"]).intersection(set(df_test["original_sample_id"]))
    if overlap:
        raise ValueError(
            f"Train/test ID overlap detected in stacking inputs: {len(overlap)} overlapping IDs"
        )

    pt_tr, pi_tr, y_tr, grp_tr, p_learned_tr = _extract_arrays(df_train)
    pt_te, pi_te, y_te, grp_te, p_learned_te = _extract_arrays(df_test)

    return {
        "train": {"pt": pt_tr, "pi": pi_tr, "y": y_tr, "grp": grp_tr, "p_learned": p_learned_tr, "df": df_train},
        "test": {"pt": pt_te, "pi": pi_te, "y": y_te, "grp": grp_te, "p_learned": p_learned_te, "df": df_test},
    }


# ═══ 2. FEATURE ENGINEERING ═══════════════════════════════════════

FEATURE_NAMES = [
    "p_text", "p_image", "p_text*p_image", "|p_text-p_image|",
    "max(pt,pi)", "min(pt,pi)", "mean(pt,pi)", "p_text^2", "p_image^2",
]


def build_meta_features(p_text, p_image):
    """Build 9-d meta-feature vector from base-learner probabilities."""
    return np.column_stack([
        p_text, p_image, p_text * p_image, np.abs(p_text - p_image),
        np.maximum(p_text, p_image), np.minimum(p_text, p_image),
        (p_text + p_image) / 2.0, p_text ** 2, p_image ** 2,
    ])


# ═══ 3. METRICS ═══════════════════════════════════════════════════

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return round(float(ece), 4)


def full_metrics(y_true, y_pred, y_prob):
    nh, h = (y_true == 0), (y_true == 1)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else None,
        "brier": round(float(brier_score_loss(y_true, np.clip(y_prob, 0, 1))), 4),
        "fpr": round(float(y_pred[nh].sum() / max(nh.sum(), 1)), 4),
        "fnr": round(float((1 - y_pred[h]).sum() / max(h.sum(), 1)), 4),
        "ece": compute_ece(y_true, y_prob),
    }


def per_group_fpr(y_true, y_pred, groups):
    results = {}
    for g in TARGET_GROUPS:
        mask = (groups == g)
        if mask.sum() == 0:
            results[g] = {"n": 0, "fpr": None}
            continue
        nh = (y_true[mask] == 0)
        fpr_val = float(y_pred[mask][nh].sum() / max(nh.sum(), 1)) if nh.sum() > 0 else None
        results[g] = {"n": int(mask.sum()), "n_non_hate": int(nh.sum()),
                       "fpr": round(fpr_val, 4) if fpr_val is not None else None}
    return results


def optimise_threshold(y_true, y_prob):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.15, 0.85, 0.005):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return round(best_t, 4), round(best_f1, 4)


def bootstrap_f1_ci(y_true, y_prob, threshold, n_boot=BOOTSTRAP_N):
    rng = np.random.default_rng(RANDOM_STATE)
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        f1s.append(f1_score(y_true[idx], (y_prob[idx] >= threshold).astype(int), zero_division=0))
    return [round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)]


# ═══ 4. PLOTS ═════════════════════════════════════════════════════

def plot_comparison(results_dict, path):
    """Bar chart comparing all models."""
    labels = list(results_dict.keys())
    keys = ["accuracy", "f1", "precision", "recall", "auc_roc", "fpr"]
    key_labels = ["Accuracy", "F1", "Precision", "Recall", "AUC-ROC", "FPR"]
    colours = ["#1D4ED8", "#F59E0B", "#8B5CF6", "#10B981", "#DC2626"][:len(labels)]

    x = np.arange(len(keys))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (mn, m) in enumerate(results_dict.items()):
        vals = [m.get(k, 0) or 0 for k in keys]
        bars = ax.bar(x + i * w, vals, w, label=mn, color=colours[i % len(colours)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels(key_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Stacking Ensemble vs Baselines (train OOF -> held-out test)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_reliability(y_true, p_uncal, p_cal, path, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    def _bin(probs):
        ms, ac = [], []
        for i in range(n_bins):
            mask = (probs > bins[i]) & (probs <= bins[i + 1])
            if mask.sum() == 0:
                ms.append((bins[i] + bins[i + 1]) / 2); ac.append(np.nan)
            else:
                ms.append(probs[mask].mean()); ac.append(y_true[mask].mean())
        return np.array(ms), np.array(ac)

    m_u, a_u = _bin(p_uncal)
    m_c, a_c = _bin(p_cal)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    mask_u, mask_c = ~np.isnan(a_u), ~np.isnan(a_c)
    ax1.plot(m_u[mask_u], a_u[mask_u], "o-", color="#DC2626", label="Uncalibrated", linewidth=2)
    ax1.plot(m_c[mask_c], a_c[mask_c], "s-", color="#10B981", label="Isotonic calibrated", linewidth=2)
    ax1.set_xlabel("Mean predicted probability"); ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Reliability Diagram", fontweight="bold"); ax1.legend(fontsize=9)
    ax2.hist(p_uncal, bins=20, alpha=0.5, color="#DC2626", label="Uncalibrated", density=True)
    ax2.hist(p_cal, bins=20, alpha=0.5, color="#10B981", label="Calibrated", density=True)
    ax2.set_xlabel("Predicted probability"); ax2.set_ylabel("Density")
    ax2.set_title("Probability Distribution", fontweight="bold"); ax2.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_group_fpr(gfpr_dict, path):
    """Per-group FPR comparison across methods."""
    methods = list(gfpr_dict.keys())
    groups = [g for g in TARGET_GROUPS
              if any(gfpr_dict[m].get(g, {}).get("fpr") is not None for m in methods)]
    x = np.arange(len(groups))
    w = 0.8 / len(methods)
    colours = ["#DC2626", "#F59E0B", "#10B981", "#1D4ED8"][:len(methods)]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, m in enumerate(methods):
        vals = [gfpr_dict[m].get(g, {}).get("fpr", 0) or 0 for g in groups]
        ax.bar(x + i * w, vals, w, label=m, color=colours[i], alpha=0.8)
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("False Positive Rate")
    ax.set_title("Per-Group FPR Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ═══ 5. MAIN ═════════════════════════════════════════════════════

def _run_one_condition(condition: str):
    cond_label = CONDITION_LABELS[condition]
    print("=" * 66)
    print(f"  STACKING ENSEMBLE — {cond_label} — Leakage-Safe Train/Test Protocol")
    print("=" * 66)
    t0 = time.time()

    # 1. Load data
    print("\n[1/7] Loading leakage-safe train/test fusion inputs ...")
    data = load_data(condition)
    tr = data["train"]
    te = data["test"]
    X_train = build_meta_features(tr["pt"], tr["pi"])
    X_test = build_meta_features(te["pt"], te["pi"])
    print(
        f"  train n={len(tr['y'])} hate_ratio={tr['y'].mean():.2%} | "
        f"test n={len(te['y'])} hate_ratio={te['y'].mean():.2%} | "
        f"meta_features={X_train.shape[1]}"
    )

    # 2. Select best meta-learner (CV on train only)
    print(f"\n[2/7] Meta-Learner Selection (CV on train only) ...")
    candidates = {
        "LogisticRegression": lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE),
        "MLP(32,16)": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
            random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    }

    # Quick selection on first 3 folds
    skf_sel = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cand_scores = {name: [] for name in candidates}
    for tr_idx, vl_idx in skf_sel.split(X_train, tr["y"]):
        for name, make_clf in candidates.items():
            clf = make_clf()
            clf.fit(X_train[tr_idx], tr["y"][tr_idx])
            p = clf.predict_proba(X_train[vl_idx])[:, 1]
            cand_scores[name].append(f1_score(tr["y"][vl_idx], (p >= 0.5).astype(int)))

    for name, scores in cand_scores.items():
        print(f"    {name:<25s} | sel-F1={np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    best_name = max(cand_scores, key=lambda k: np.mean(cand_scores[k]))
    print(f"  Best: {best_name}")

    # 3. Build train OOF predictions for diagnostics and threshold tuning
    print(f"\n[3/7] Building train OOF predictions with {best_name} ...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_train_proba = np.zeros(len(tr["y"]), dtype=float)
    fold_f1s = []

    for fold_i, (tr_idx, vl_idx) in enumerate(skf.split(X_train, tr["y"]), 1):
        t1 = time.time()
        clf = candidates[best_name]()
        clf.fit(X_train[tr_idx], tr["y"][tr_idx])
        oof_train_proba[vl_idx] = clf.predict_proba(X_train[vl_idx])[:, 1]
        fold_f1 = f1_score(tr["y"][vl_idx], (oof_train_proba[vl_idx] >= 0.5).astype(int))
        fold_f1s.append(fold_f1)
        print(f"    Fold {fold_i}: F1={fold_f1:.4f}  ({time.time()-t1:.1f}s)")

    # Optimise threshold on train OOF predictions
    best_thresh, _ = optimise_threshold(tr["y"], oof_train_proba)
    oof_train_preds = (oof_train_proba >= best_thresh).astype(int)
    oof_train_m = full_metrics(tr["y"], oof_train_preds, oof_train_proba)
    print(f"  Train OOF threshold={best_thresh}  F1={oof_train_m['f1']}  AUC={oof_train_m['auc_roc']}")
    print(f"  Fold F1: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")

    # Fit final model on full train, evaluate once on held-out test
    final_clf = candidates[best_name]()
    final_clf.fit(X_train, tr["y"])
    test_proba = final_clf.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_thresh).astype(int)
    stacking_m = full_metrics(te["y"], test_preds, test_proba)
    print(f"  Test F1={stacking_m['f1']}  AUC={stacking_m['auc_roc']}  FPR={stacking_m['fpr']}")

    gfpr_stack = per_group_fpr(te["y"], test_preds, te["grp"])

    # Bootstrap CI on held-out test
    f1_ci = bootstrap_f1_ci(te["y"], test_proba, best_thresh)
    print(f"  F1 95% CI: [{f1_ci[0]}, {f1_ci[1]}]")

    # 4. Isotonic calibration (fit on train OOF, apply to test)
    print(f"\n[4/7] Isotonic calibration (train-fit, test-apply) ...")
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(oof_train_proba, tr["y"])
    oof_cal_train = iso.predict(oof_train_proba)
    test_cal = iso.predict(test_proba)
    cal_thresh, _ = optimise_threshold(tr["y"], oof_cal_train)
    test_cal_preds = (test_cal >= cal_thresh).astype(int)
    cal_m = full_metrics(te["y"], test_cal_preds, test_cal)
    print(f"  Calibrated: F1={cal_m['f1']}  ECE={cal_m['ece']}  (uncal ECE={stacking_m['ece']})")

    # 5. Compare with baselines (thresholds tuned on train)
    print(f"\n[5/7] Comparing with baselines ...")

    # Scalar fusion baseline
    t_learned, _ = optimise_threshold(tr["y"], tr["p_learned"])
    pred_learned = (te["p_learned"] >= t_learned).astype(int)
    fusion_m = full_metrics(te["y"], pred_learned, te["p_learned"])
    gfpr_fusion = per_group_fpr(te["y"], pred_learned, te["grp"])

    # Text-only
    text_thresh, _ = optimise_threshold(tr["y"], tr["pt"])
    text_preds = (te["pt"] >= text_thresh).astype(int)
    text_m = full_metrics(te["y"], text_preds, te["pt"])
    gfpr_text = per_group_fpr(te["y"], text_preds, te["grp"])

    # Image-only
    img_thresh, _ = optimise_threshold(tr["y"], tr["pi"])
    img_preds = (te["pi"] >= img_thresh).astype(int)
    image_m = full_metrics(te["y"], img_preds, te["pi"])

    all_models = {
        "Text-Only": text_m, "Image-Only": image_m,
        "Scalar Fusion": fusion_m,
        "Stacking (uncal)": stacking_m, "Stacking (cal)": cal_m,
    }

    print(f"\n  {'Model':<25s}  {'F1':>7}  {'AUC':>7}  {'FPR':>7}  {'ECE':>7}")
    print(f"  {'-'*58}")
    for label, m in all_models.items():
        auc_s = f"{m['auc_roc']:.4f}" if m.get("auc_roc") else "  --  "
        print(f"  {label:<25s}  {m['f1']:>7.4f}  {auc_s:>7}  {m['fpr']:>7.4f}  {m['ece']:>7.4f}")

    # 6. Plots & output
    print(f"\n[6/7] Generating plots & saving results ...")
    plot_comparison(all_models, PLOTS_DIR / "stacking_comparison.png")
    plot_reliability(te["y"], test_proba, test_cal, PLOTS_DIR / "stacking_reliability.png")
    plot_group_fpr({"Text-Only": gfpr_text, "Scalar Fusion": gfpr_fusion,
                    "Stacking": gfpr_stack}, PLOTS_DIR / "stacking_per_group_fpr.png")
    print("  Plots saved.")

    # Predictions CSVs
    pred_out = te["df"][["original_sample_id", "counterfactual_id", "text",
                         "class_label", "target_group"]].copy()
    pred_out["y_true"] = te["y"]
    pred_out["p_text"] = np.round(te["pt"], 6)
    pred_out["p_image"] = np.round(te["pi"], 6)
    pred_out["p_stacking_test"] = np.round(test_proba, 6)
    pred_out["p_stacking_cal_test"] = np.round(test_cal, 6)
    pred_out["pred_stacking"] = test_preds
    pred_out["pred_stacking_cal"] = test_cal_preds
    pred_out.to_csv(PRED_DIR / f"stacking_predictions_test_{condition}.csv", index=False)

    train_oof_out = tr["df"][["original_sample_id", "counterfactual_id", "text",
                               "class_label", "target_group"]].copy()
    train_oof_out["y_true"] = tr["y"]
    train_oof_out["p_text"] = np.round(tr["pt"], 6)
    train_oof_out["p_image"] = np.round(tr["pi"], 6)
    train_oof_out["p_stacking_oof_train"] = np.round(oof_train_proba, 6)
    train_oof_out["p_stacking_cal_oof_train"] = np.round(oof_cal_train, 6)
    train_oof_out["pred_stacking_oof_train"] = oof_train_preds
    train_oof_out.to_csv(PRED_DIR / f"stacking_train_oof_predictions_{condition}.csv", index=False)

    # JSON
    output = {
        "condition": condition,
        "condition_label": cond_label,
        "description": "Stacking Ensemble: train OOF fitting + single held-out test evaluation",
        "protocol": "5-fold CV/OOF on train only; final meta-learner fit on full train; one-shot evaluation on test",
        "best_meta_learner": best_name,
        "n_train_samples": int(len(tr["y"])),
        "n_test_samples": int(len(te["y"])),
        "n_folds": N_FOLDS,
        "n_meta_features": int(X_train.shape[1]),
        "feature_names": FEATURE_NAMES,
        "meta_learner_selection": {k: {"mean_f1": round(np.mean(v), 4), "std_f1": round(np.std(v), 4)}
                                   for k, v in cand_scores.items()},
        "threshold_train_oof": best_thresh,
        "threshold_calibrated_train_oof": cal_thresh,
        "fold_f1s": [round(f, 4) for f in fold_f1s],
        "oof_metrics_uncalibrated": oof_train_m,
        "test_metrics_uncalibrated": stacking_m,
        "test_metrics_calibrated": cal_m,
        "per_group_fpr_stacking_test": gfpr_stack,
        "f1_95ci": f1_ci,
        "baselines": {"text_only": text_m, "image_only": image_m, "scalar_fusion": fusion_m},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / f"stacking_ensemble_results_{condition}.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    delta = stacking_m["f1"] - fusion_m["f1"]
    print(f"\n{'='*66}")
    print(f"  STACKING ENSEMBLE SUMMARY (train OOF -> held-out test)")
    print(f"{'='*66}")
    print(f"  Meta-learner : {best_name}")
    print(f"  Train OOF F1 : {oof_train_m['f1']:.4f}")
    print(f"  Test F1      : {stacking_m['f1']:.4f}  CI=[{f1_ci[0]}, {f1_ci[1]}]")
    print(f"  Test F1 cal  : {cal_m['f1']:.4f}")
    print(f"  Test AUC-ROC : {stacking_m['auc_roc']}")
    print(f"  ECE uncal/cal: {stacking_m['ece']} / {cal_m['ece']}")
    print(f"  vs Scalar F1 : {fusion_m['f1']:.4f}  (Delta={delta:+.4f})")
    print(f"{'='*66}")
    return output


def main():
    aggregated = {}
    for condition in FUSION_CONDITIONS:
        aggregated[condition] = _run_one_condition(condition)

    consolidated_rows = []
    for condition in FUSION_CONDITIONS:
        row = {
            "condition": CONDITION_LABELS[condition],
            "f1": aggregated[condition]["test_metrics_uncalibrated"]["f1"],
            "auc_roc": aggregated[condition]["test_metrics_uncalibrated"].get("auc_roc"),
            "fpr": aggregated[condition]["test_metrics_uncalibrated"]["fpr"],
            "ece": aggregated[condition]["test_metrics_uncalibrated"]["ece"],
            "best_meta_learner": aggregated[condition]["best_meta_learner"],
        }
        consolidated_rows.append(row)

    output = {
        "description": "Stacking ensemble across nCF / CF-no-adv / CF+GRL fusion conditions",
        "conditions": list(FUSION_CONDITIONS),
        "condition_labels": CONDITION_LABELS,
        "summary_table": consolidated_rows,
        "by_condition": aggregated,
    }

    with open(OUT_DIR / "stacking_ensemble_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nConsolidated stacking results saved: stacking_ensemble_results.json")
    return output


if __name__ == "__main__":
    main()
