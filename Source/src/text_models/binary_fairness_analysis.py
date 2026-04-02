"""
binary_fairness_analysis.py
============================
End-to-end binary classification evaluation of Counterfactual Data Augmentation
(CAD) for hate speech detection.

Task        : Binary classification — hate (1) vs non-hate (0)
Conditions  : nCF (6k originals) vs CF (18k with counterfactuals)
Models      : Logistic Regression, Ridge Regression, Naive Bayes, Random Forest
Metrics     : Accuracy, Precision, Recall, F1, AUC-ROC, FPR, FNR, Brier Score
              Bootstrap 95% CI for all key metrics (n=2000)
Statistics  : McNemar's test — nCF vs CF error pattern comparison (per model)
              DeLong test — AUC comparison between conditions (per model)
Plots       : ROC curves, calibration, confusion matrices, overall metric
              comparison, FPR/FNR delta bar chart, PR curves, summary heatmap
Export      : binary_fairness_results.json  (all numeric results)

Author      : Senior ML / Fairness-in-ML pipeline
"""

# ─── Standard library ────────────────────────────────────────────────────────
import os, sys, json, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, brier_score_loss,
    precision_recall_curve, average_precision_score, roc_auc_score,
)
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from canonical_splits import get_canonical_splits, build_condition_split_frames  # noqa: E402

DATA_6K      = PROJECT_ROOT / "src" / "counterfactual_gen" / "hate_speech_dataset_6k.csv"
DATA_18K     = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k.csv"
OUT_DIR      = Path(__file__).resolve().parent / "binary_fairness_results"
MODELS_DIR   = OUT_DIR / "models"
PLOTS_DIR    = OUT_DIR / "plots"
for _d in (OUT_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
BOOTSTRAP_N  = 2000
BOOTSTRAP_CI = 0.95

PALETTE = {"ncf": "#2563EB", "cf": "#DC2626"}

MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "ridge_regression":    "Ridge Regression",
    "naive_bayes":         "Naive Bayes",
    "random_forest":       "Random Forest",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                     "figure.facecolor": "white"})


# ==============================================================================
# 1.  DATA LOADING & SPLITTING
# ==============================================================================

def _filter_non_english(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    mask = df["text"].fillna("").apply(
        lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1) > threshold
    )
    n = mask.sum()
    if n:
        print(f"  Dropped {n} non-English rows (>{threshold*100:.0f}% non-ASCII).")
    return df[~mask].copy()


def _binary_label(polarity: str) -> int:
    return 1 if str(polarity).strip().lower() == "hate" else 0


def load_condition(condition: str) -> dict:
    """
    Load & split data using CANONICAL splits (class_label-stratified 70/15/15).

        nCF : train/val/test on 6k original samples only.
        CF  : train augmented with all CF variants and val is CF-augmented for
            threshold tuning; test remains originals only.

    Critical: uses get_canonical_splits() so val/test IDs are identical
    across all pipelines (text, image, cross-modal).
    """
    print(f"\n{'─'*60}")
    print(f"  Condition: {condition.upper()}")
    print(f"{'─'*60}")

    if condition == "ncf":
        df = pd.read_csv(DATA_6K)
        df = _filter_non_english(df)
        df = df.rename(columns={"sample_id": "original_sample_id"})
        df["cf_type"] = "original"
    else:
        df = pd.read_csv(DATA_18K)
        df = _filter_non_english(df)

    df["binary_label"] = df["polarity"].apply(_binary_label)

    originals = df[df["cf_type"] == "original"].copy()
    print(f"  Originals: {len(originals):,}  |  "
          f"Hate: {originals['binary_label'].sum():,}  "
          f"Non-hate: {(originals['binary_label']==0).sum():,}")

    # ── Use canonical splits (class_label-stratified) ─────────────────────
    _canon = get_canonical_splits()
    train_ids = _canon["train_ids"]
    val_ids   = _canon["val_ids"]
    test_ids  = _canon["test_ids"]

    split_frames = build_condition_split_frames(
        df=df,
        condition=condition,
        splits=_canon,
        augment_val_for_cf=True,
    )
    train_df = split_frames["train"]
    val_df = split_frames["val"]
    test_df = split_frames["test"]

    # Leakage guard
    assert train_ids.isdisjoint(val_ids),  "LEAKAGE: train ∩ val!"
    assert train_ids.isdisjoint(test_ids), "LEAKAGE: train ∩ test!"
    assert val_ids.isdisjoint(test_ids),   "LEAKAGE: val ∩ test!"

    print(f"  Train: {len(train_df):,} ({train_df['binary_label'].mean()*100:.1f}% hate) "
          f"| Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  No data leakage confirmed.")
    return dict(train=train_df, val=val_df, test=test_df, condition=condition)


# ==============================================================================
# 2.  FEATURE EXTRACTION — TF-IDF
# ==============================================================================

def build_features(splits: dict, vectorizer=None) -> dict:
    """Fit TF-IDF on train only. Pass pre-fitted vectorizer for CF condition."""
    cond = splits["condition"]
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=15_000, ngram_range=(1, 3),
            sublinear_tf=True, strip_accents="unicode",
            analyzer="word", token_pattern=r"\b\w+\b",
            stop_words="english", min_df=2,
        )
        vectorizer.fit(splits["train"]["text"].fillna(""))
        joblib.dump(vectorizer, MODELS_DIR / f"tfidf_{cond}.joblib")
        n_feat = vectorizer.transform(["x"]).shape[1]
        print(f"  TF-IDF fitted: {n_feat:,} features")

    def _t(df):
        return (vectorizer.transform(df["text"].fillna("")),
                df["binary_label"].values)

    X_tr, y_tr = _t(splits["train"])
    X_v,  y_v  = _t(splits["val"])
    X_te, y_te = _t(splits["test"])

    return dict(X_train=X_tr, y_train=y_tr,
                X_val=X_v,   y_val=y_v,
                X_test=X_te, y_test=y_te,
                condition=cond, vectorizer=vectorizer,
                test_df=splits["test"])


# ==============================================================================
# 3.  MODEL DEFINITIONS
# ==============================================================================

def get_models() -> dict:
    """
    Four complementary classifiers for binary hate/non-hate classification.

    Logistic Regression  — strong linear baseline with L2 regularisation.
    Ridge Regression     — bias-variance tradeoff via L2 penalty; fast and
                           interpretable; calibrated for probability output.
    Naive Bayes          — generative probabilistic baseline; excels on high-dim
                           sparse TF-IDF; MultinomialNB works with TF-IDF >= 0.
    Random Forest        — non-linear ensemble; captures feature interactions;
                           robust to high-dimensional input.
    """
    lr = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    ridge_base = RidgeClassifier(alpha=1.0, class_weight="balanced")
    # RidgeClassifier has no predict_proba — wrap with isotonic calibration
    ridge = CalibratedClassifierCV(ridge_base, cv=5, method="isotonic")

    nb = MultinomialNB(alpha=0.1)   # TF-IDF values >= 0, compatible with MNB

    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )
    return {
        "logistic_regression": lr,
        "ridge_regression":    ridge,
        "naive_bayes":         nb,
        "random_forest":       rf,
    }


# ==============================================================================
# 4.  TRAINING + EVALUATION
# ==============================================================================

def _safe_div(num, den):
    return float(num / den) if den > 0 else float("nan")


def train_and_evaluate(model, name: str, feats: dict) -> dict:
    cond = feats["condition"]
    print(f"\n  >> {MODEL_DISPLAY[name]} [{cond.upper()}]", end=" ... ", flush=True)
    t0 = time.time()
    model.fit(feats["X_train"], feats["y_train"])
    elapsed = time.time() - t0
    print(f"done ({elapsed:.2f}s)")

    y_true = feats["y_test"]
    y_pred = model.predict(feats["X_test"])
    y_prob = model.predict_proba(feats["X_test"])[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred, zero_division=0),
        f1        = f1_score(y_true, y_pred, zero_division=0),
        roc_auc   = roc_auc_score(y_true, y_prob),
        avg_prec  = average_precision_score(y_true, y_prob),
        brier     = brier_score_loss(y_true, y_prob),
        fpr       = _safe_div(fp, fp + tn),
        fnr       = _safe_div(fn, fn + tp),
        tpr       = _safe_div(tp, tp + fn),
        tnr       = _safe_div(tn, tn + fp),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )

    # Bootstrap CI (95%) for 5 key metrics
    rng = np.random.default_rng(RANDOM_STATE)
    metrics["ci"] = {}
    idx = np.arange(len(y_true))
    for met_name, fn_ in [
        ("accuracy", lambda yt, yp, ypr: accuracy_score(yt, yp)),
        ("f1",       lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0)),
        ("roc_auc",  lambda yt, yp, ypr: roc_auc_score(yt, ypr)),
        ("fpr",      lambda yt, yp, ypr: _safe_div(
                         int(confusion_matrix(yt, yp).ravel()[1]),
                         int(confusion_matrix(yt, yp).ravel()[1]) +
                         int(confusion_matrix(yt, yp).ravel()[0]))),
        ("fnr",      lambda yt, yp, ypr: _safe_div(
                         int(confusion_matrix(yt, yp).ravel()[2]),
                         int(confusion_matrix(yt, yp).ravel()[2]) +
                         int(confusion_matrix(yt, yp).ravel()[3]))),
    ]:
        vals = []
        for _ in range(BOOTSTRAP_N):
            s = rng.choice(idx, size=len(idx), replace=True)
            try:
                v = fn_(y_true[s], y_pred[s], y_prob[s])
                if not np.isnan(v):
                    vals.append(v)
            except Exception:
                pass
        alpha = (1 - BOOTSTRAP_CI) / 2
        metrics["ci"][met_name] = (
            float(np.quantile(vals, alpha)),
            float(np.quantile(vals, 1 - alpha)),
        ) if vals else (float("nan"), float("nan"))

    print(f"    Acc={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}  "
          f"AUC={metrics['roc_auc']:.4f}  "
          f"FPR={metrics['fpr']:.4f}  FNR={metrics['fnr']:.4f}")

    joblib.dump(model, MODELS_DIR / f"{name}_{cond}.joblib")

    return dict(
        name=name, condition=cond, model=model,
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        training_time=elapsed, **metrics,
    )


# ==============================================================================
# 5.  STATISTICAL TESTS
# ==============================================================================

def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """McNemar's test (continuity-corrected): are nCF and CF error patterns different?"""
    b = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    c = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    if b + c < 10:
        return dict(statistic=None, p_value=None, b=b, c=c, note="n<10, unreliable")
    stat  = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = float(1 - stats.chi2.cdf(stat, df=1))
    return dict(statistic=float(stat), p_value=p_val, b=b, c=c)


def delong_auc_test(y_true, y_prob_a, y_prob_b) -> dict:
    """DeLong's test for correlated ROC curves (same test set)."""
    def _wilcoxon_var(y_true, y_prob):
        pos = y_prob[y_true == 1]
        neg = y_prob[y_true == 0]
        n1, n0 = len(pos), len(neg)
        if n1 == 0 or n0 == 0:
            return float("nan"), float("nan"), (None, None)
        pv1 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        pv0 = np.array([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])
        auc_ = np.mean(pv1)
        var_ = (np.var(pv1, ddof=1) / n1 + np.var(pv0, ddof=1) / n0)
        return auc_, var_, (pv1, pv0)

    auc_a, var_a, pv_a = _wilcoxon_var(y_true, y_prob_a)
    auc_b, var_b, pv_b = _wilcoxon_var(y_true, y_prob_b)

    if pv_a[0] is None or pv_b[0] is None:
        return dict(z=float("nan"), p_value=float("nan"),
                    auc_a=float(auc_a), auc_b=float(auc_b))

    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())
    cov = (np.cov(pv_a[0], pv_b[0])[0, 1] / n1 +
           np.cov(pv_a[1], pv_b[1])[0, 1] / n0)

    se = float(np.sqrt(var_a + var_b - 2 * cov))
    if se == 0:
        return dict(z=float("nan"), p_value=float("nan"),
                    auc_a=float(auc_a), auc_b=float(auc_b))

    z     = float((auc_a - auc_b) / se)
    p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
    return dict(z=z, p_value=p_val, auc_a=float(auc_a), auc_b=float(auc_b))


# ==============================================================================
# 6.  PLOTS
# ==============================================================================

def plot_roc_curves(all_results: dict, out: Path):
    model_names = list(MODEL_DISPLAY.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    ls = {"ncf": "--", "cf": "-"}

    for ax, m in zip(axes, model_names):
        for cond in ["ncf", "cf"]:
            key = f"{m}_{cond}"
            if key not in all_results:
                continue
            r = all_results[key]
            fpr_r, tpr_r, _ = roc_curve(r["y_true"], r["y_prob"])
            ax.plot(fpr_r, tpr_r, lw=2, color=PALETTE[cond], ls=ls[cond],
                    label=f"{cond.upper()} (AUC={r['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(MODEL_DISPLAY[m], fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("ROC Curves — Binary Hate Detection (nCF vs CF)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out / "roc_curves.png")
    plt.close(fig)
    print("  Saved: roc_curves.png")


def plot_pr_curves(all_results: dict, out: Path):
    model_names = list(MODEL_DISPLAY.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    ls = {"ncf": "--", "cf": "-"}

    for ax, m in zip(axes, model_names):
        for cond in ["ncf", "cf"]:
            key = f"{m}_{cond}"
            if key not in all_results:
                continue
            r = all_results[key]
            prec_r, rec_r, _ = precision_recall_curve(r["y_true"], r["y_prob"])
            ap = r["avg_prec"]
            ax.plot(rec_r, prec_r, lw=2, color=PALETTE[cond], ls=ls[cond],
                    label=f"{cond.upper()} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(MODEL_DISPLAY[m], fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Precision-Recall Curves — Binary Hate Detection (nCF vs CF)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out / "pr_curves.png")
    plt.close(fig)
    print("  Saved: pr_curves.png")


def plot_calibration(all_results: dict, out: Path):
    model_names = list(MODEL_DISPLAY.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    ls = {"ncf": "--", "cf": "-"}

    for ax, m in zip(axes, model_names):
        for cond in ["ncf", "cf"]:
            key = f"{m}_{cond}"
            if key not in all_results:
                continue
            r = all_results[key]
            prob_true, prob_pred = calibration_curve(
                r["y_true"], r["y_prob"], n_bins=10, strategy="uniform")
            ax.plot(prob_pred, prob_true, marker="o", lw=2,
                    color=PALETTE[cond], ls=ls[cond],
                    label=f"{cond.upper()} (Brier={r['brier']:.4f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(MODEL_DISPLAY[m], fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Calibration Plots — Binary Hate Detection (nCF vs CF)",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out / "calibration.png")
    plt.close(fig)
    print("  Saved: calibration.png")


def plot_confusion_matrices(all_results: dict, out: Path):
    model_names = list(MODEL_DISPLAY.keys())
    conditions  = ["ncf", "cf"]
    fig, axes = plt.subplots(len(model_names), len(conditions),
                             figsize=(10, 4.5 * len(model_names)))

    for r_i, m in enumerate(model_names):
        for c_i, cond in enumerate(conditions):
            ax  = axes[r_i][c_i]
            key = f"{m}_{cond}"
            if key not in all_results:
                ax.axis("off")
                continue
            r  = all_results[key]
            cm = confusion_matrix(r["y_true"], r["y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Hate", "Hate"],
                        yticklabels=["Non-Hate", "Hate"],
                        ax=ax, cbar=False, linewidths=0.5)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(
                f"{MODEL_DISPLAY[m]} [{cond.upper()}]  "
                f"Acc={r['accuracy']:.3f}  F1={r['f1']:.3f}",
                fontweight="bold", fontsize=10,
            )

    fig.suptitle("Confusion Matrices — Binary Classification (nCF vs CF)",
                 fontweight="bold", fontsize=13, y=1.005)
    plt.tight_layout()
    fig.savefig(out / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: confusion_matrices.png")


def plot_metrics_comparison(all_results: dict, out: Path):
    """Four-panel bar chart with 95% CI error bars."""
    model_names = list(MODEL_DISPLAY.keys())
    met_list = [
        ("accuracy", "Accuracy"),
        ("f1",       "F1 Score"),
        ("roc_auc",  "AUC-ROC"),
        ("fpr",      "FPR (False Positive Rate)"),
    ]
    x = np.arange(len(model_names))
    w = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (met, label) in zip(axes, met_list):
        ncf_vals, ncf_lo, ncf_hi = [], [], []
        cf_vals,  cf_lo,  cf_hi  = [], [], []

        for m in model_names:
            for vals_list, lo_list, hi_list, cond in [
                (ncf_vals, ncf_lo, ncf_hi, "ncf"),
                (cf_vals,  cf_lo,  cf_hi,  "cf"),
            ]:
                r  = all_results.get(f"{m}_{cond}", {})
                v  = r.get(met, float("nan"))
                ci = r.get("ci", {}).get(met, (float("nan"), float("nan")))
                vals_list.append(v)
                lo_list.append(abs(v - ci[0]) if not np.isnan(ci[0]) else 0)
                hi_list.append(abs(ci[1] - v) if not np.isnan(ci[1]) else 0)

        b1 = ax.bar(x - w/2, ncf_vals, w, label="nCF (6k)",
                    color=PALETTE["ncf"], alpha=0.85,
                    yerr=[ncf_lo, ncf_hi],
                    error_kw=dict(ecolor="#374151", capsize=5, lw=1.5))
        b2 = ax.bar(x + w/2, cf_vals, w, label="CF (18k)",
                    color=PALETTE["cf"], alpha=0.85,
                    yerr=[cf_lo, cf_hi],
                    error_kw=dict(ecolor="#374151", capsize=5, lw=1.5))

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in model_names],
                           rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + max(bar.get_height() * 0.01, 0.005),
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(
        "Binary Classification Performance — nCF (6k) vs CF (18k)\n"
        "Error bars = Bootstrap 95% CI  (n=2000 resamples)",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(out / "metrics_comparison.png")
    plt.close(fig)
    print("  Saved: metrics_comparison.png")


def plot_fpr_fnr_delta(all_results: dict, out: Path):
    """Delta FPR and FNR (CF - nCF) per model."""
    model_names   = list(MODEL_DISPLAY.keys())
    display_names = [MODEL_DISPLAY[m] for m in model_names]

    delta_fpr, delta_fnr = [], []
    ci_fpr_lo, ci_fpr_hi = [], []
    ci_fnr_lo, ci_fnr_hi = [], []

    for m in model_names:
        ncf = all_results.get(f"{m}_ncf", {})
        cf  = all_results.get(f"{m}_cf",  {})
        d_fpr = cf.get("fpr", float("nan")) - ncf.get("fpr", float("nan"))
        d_fnr = cf.get("fnr", float("nan")) - ncf.get("fnr", float("nan"))
        delta_fpr.append(d_fpr)
        delta_fnr.append(d_fnr)

        ncf_ci_fpr = ncf.get("ci", {}).get("fpr", (float("nan"), float("nan")))
        cf_ci_fpr  = cf.get("ci",  {}).get("fpr", (float("nan"), float("nan")))
        ncf_ci_fnr = ncf.get("ci", {}).get("fnr", (float("nan"), float("nan")))
        cf_ci_fnr  = cf.get("ci",  {}).get("fnr", (float("nan"), float("nan")))

        ci_fpr_lo.append(max(d_fpr - (cf_ci_fpr[0] - ncf_ci_fpr[1]), 0)
                         if not any(np.isnan([cf_ci_fpr[0], ncf_ci_fpr[1]])) else 0)
        ci_fpr_hi.append(max((cf_ci_fpr[1] - ncf_ci_fpr[0]) - d_fpr, 0)
                         if not any(np.isnan([cf_ci_fpr[1], ncf_ci_fpr[0]])) else 0)
        ci_fnr_lo.append(max(d_fnr - (cf_ci_fnr[0] - ncf_ci_fnr[1]), 0)
                         if not any(np.isnan([cf_ci_fnr[0], ncf_ci_fnr[1]])) else 0)
        ci_fnr_hi.append(max((cf_ci_fnr[1] - ncf_ci_fnr[0]) - d_fnr, 0)
                         if not any(np.isnan([cf_ci_fnr[1], ncf_ci_fnr[0]])) else 0)

    x = np.arange(len(model_names))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, deltas, lo, hi, label, note in [
        (axes[0], delta_fpr, ci_fpr_lo, ci_fpr_hi,
         "Delta FPR  (CF - nCF)",
         "Positive = more non-hate falsely flagged after CAD"),
        (axes[1], delta_fnr, ci_fnr_lo, ci_fnr_hi,
         "Delta FNR  (CF - nCF)",
         "Negative = fewer hate samples missed after CAD"),
    ]:
        colours = ["#DC2626" if d > 0.005 else
                   "#2563EB" if d < -0.005 else
                   "#9CA3AF" for d in deltas]
        bars = ax.bar(x, deltas, w, color=colours, alpha=0.85,
                      yerr=[lo, hi],
                      error_kw=dict(ecolor="#374151", capsize=6, lw=1.5))
        ax.axhline(0, color="#374151", lw=1.2, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label}\n{note}", fontweight="bold", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.002 if h >= 0 else -0.008),
                    f"{h:+.4f}", ha="center",
                    va="bottom" if h >= 0 else "top", fontsize=9)

    fig.suptitle(
        "FPR / FNR Change After Counterfactual Augmentation (CF - nCF)\n"
        "Error bars = propagated Bootstrap 95% CI",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out / "fpr_fnr_delta.png")
    plt.close(fig)
    print("  Saved: fpr_fnr_delta.png")


def plot_summary_heatmap(all_results: dict, out: Path):
    """Heatmap of all key metrics across model x condition."""
    model_names   = list(MODEL_DISPLAY.keys())
    metric_names  = ["accuracy", "precision", "recall", "f1",
                     "roc_auc", "fpr", "fnr", "brier"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1",
                     "AUC-ROC", "FPR", "FNR", "Brier"]

    rows, row_labels = [], []
    for m in model_names:
        for cond in ["ncf", "cf"]:
            key = f"{m}_{cond}"
            if key not in all_results:
                continue
            r = all_results[key]
            rows.append([r.get(mn, float("nan")) for mn in metric_names])
            row_labels.append(f"{MODEL_DISPLAY[m]}\n[{cond.upper()}]")

    data = pd.DataFrame(rows, columns=metric_labels, index=row_labels)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(data.astype(float), annot=True, fmt=".3f",
                cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Metric value"})
    ax.set_title(
        "Binary Classification Metrics Heatmap — All Models x Conditions\n"
        "Green = higher; FPR/FNR/Brier: lower is better",
        fontweight="bold", pad=14,
    )
    plt.tight_layout()
    fig.savefig(out / "metrics_heatmap.png")
    plt.close(fig)
    print("  Saved: metrics_heatmap.png")


# ==============================================================================
# 7.  RESULTS PRINTING
# ==============================================================================

def print_summary(all_results: dict, mcnemar: dict, delong: dict):
    SEP = "=" * 100

    print(f"\n{SEP}")
    print("  BINARY CLASSIFICATION RESULTS — nCF (6k) vs CF (18k)")
    print(SEP)
    hdr = (f"{'Model':<22} {'Cond':<6} {'Acc':>8} {'95% CI':>14} "
           f"{'F1':>8} {'95% CI':>14} {'AUC':>8} {'FPR':>8} {'FNR':>8} "
           f"{'Brier':>8} {'Time':>6}")
    print(hdr)
    print("-" * 100)
    for m in MODEL_DISPLAY:
        for cond in ["ncf", "cf"]:
            key = f"{m}_{cond}"
            if key not in all_results:
                continue
            r    = all_results[key]
            ci_a = r["ci"]["accuracy"]
            ci_f = r["ci"]["f1"]
            print(
                f"{MODEL_DISPLAY[m]:<22} {cond.upper():<6} "
                f"{r['accuracy']:>8.4f} [{ci_a[0]:.3f}-{ci_a[1]:.3f}] "
                f"{r['f1']:>8.4f} [{ci_f[0]:.3f}-{ci_f[1]:.3f}] "
                f"{r['roc_auc']:>8.4f} {r['fpr']:>8.4f} {r['fnr']:>8.4f} "
                f"{r['brier']:>8.4f} {r['training_time']:>5.2f}s"
            )
        print()

    print(f"\n{SEP}")
    print("  FPR / FNR DELTA  (CF - nCF)")
    print("  Delta FPR > 0 = CAD increased false positive rate")
    print("  Delta FNR < 0 = CAD reduced false negative rate (improvement)")
    print(SEP)
    hdr2 = (f"{'Model':<22} {'nCF FPR':>9} {'CF FPR':>8} {'Delta FPR':>10} "
            f"{'nCF FNR':>9} {'CF FNR':>8} {'Delta FNR':>10} {'AUC Delta':>10}")
    print(hdr2)
    print("-" * 90)
    for m in MODEL_DISPLAY:
        ncf = all_results.get(f"{m}_ncf", {})
        cf  = all_results.get(f"{m}_cf",  {})
        d_fpr = cf.get("fpr", float("nan")) - ncf.get("fpr", float("nan"))
        d_fnr = cf.get("fnr", float("nan")) - ncf.get("fnr", float("nan"))
        d_auc = cf.get("roc_auc", float("nan")) - ncf.get("roc_auc", float("nan"))
        print(
            f"{MODEL_DISPLAY[m]:<22} "
            f"{ncf.get('fpr', float('nan')):>9.4f} {cf.get('fpr', float('nan')):>8.4f} {d_fpr:>+10.4f} "
            f"{ncf.get('fnr', float('nan')):>9.4f} {cf.get('fnr', float('nan')):>8.4f} {d_fnr:>+10.4f} "
            f"{d_auc:>+10.4f}"
        )

    print(f"\n{SEP}")
    print("  STATISTICAL TESTS  (nCF vs CF per model, same test set)")
    print("  McNemar: are error patterns significantly different?")
    print("  DeLong:  are ROC curves significantly different?")
    print(SEP)
    hdr3 = (f"{'Model':<22} {'McNemar chi2':>13} {'p':>9} {'sig':>5} "
            f"|  {'DeLong z':>10} {'p':>9} {'sig':>5} "
            f"| {'AUC nCF':>9} {'AUC CF':>9}")
    print(hdr3)
    print("-" * 98)
    for m in MODEL_DISPLAY:
        mn = mcnemar.get(m, {})
        dl = delong.get(m, {})

        def _sig(p):
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return "n/a"
            return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        mn_stat = f"{mn['statistic']:.3f}" if mn.get("statistic") else "n/a"
        mn_p    = f"{mn['p_value']:.4f}"   if mn.get("p_value")   else "n/a"
        dl_z    = f"{dl.get('z', float('nan')):.3f}"
        dl_p    = f"{dl['p_value']:.4f}"   if dl.get("p_value")   else "n/a"

        print(
            f"{MODEL_DISPLAY[m]:<22} {mn_stat:>13} {mn_p:>9} {_sig(mn.get('p_value')):>5} "
            f"|  {dl_z:>10} {dl_p:>9} {_sig(dl.get('p_value')):>5} "
            f"| {dl.get('auc_a', float('nan')):>9.4f} {dl.get('auc_b', float('nan')):>9.4f}"
        )


# ==============================================================================
# 8.  JSON EXPORT
# ==============================================================================

def _to_json(obj):
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return _to_json(obj.tolist())
    return obj


def export_json(all_results: dict, mcnemar: dict, delong: dict, out: Path):
    exclude = {"model", "y_true", "y_pred", "y_prob", "test_df"}
    payload = {
        "overall_metrics": {
            k: _to_json({kk: vv for kk, vv in v.items() if kk not in exclude})
            for k, v in all_results.items()
        },
        "mcnemar_tests": _to_json(mcnemar),
        "delong_tests":  _to_json(delong),
    }
    path = out / "binary_fairness_results.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results exported -> {path}")


# ==============================================================================
# 9.  MAIN PIPELINE
# ==============================================================================

def main():
    print("=" * 70)
    print("  BINARY CLASSIFICATION ANALYSIS — CAD in Hate Speech Detection")
    print("  Models : LR  |  Ridge  |  Naive Bayes  |  Random Forest")
    print("  Conditions : nCF (6k originals)  vs  CF (18k augmented)")
    print("=" * 70)

    print("\n>> STEP 1  Data Loading & Splitting")
    ncf_splits = load_condition("ncf")
    cf_splits  = load_condition("cf")

    print("\n>> STEP 2  TF-IDF Feature Extraction")
    ncf_feat = build_features(ncf_splits)
    cf_feat  = build_features(cf_splits, vectorizer=ncf_feat["vectorizer"])

    print("\n>> STEP 3  Model Training & Evaluation")
    all_results = {}
    models      = get_models()

    for name, model in models.items():
        print(f"\n  -- {MODEL_DISPLAY[name]} --")
        all_results[f"{name}_ncf"] = train_and_evaluate(clone(model), name, ncf_feat)
        all_results[f"{name}_cf"]  = train_and_evaluate(clone(model), name, cf_feat)

    print("\n>> STEP 4  Statistical Tests (McNemar + DeLong)")
    mcnemar_results, delong_results = {}, {}
    for name in MODEL_DISPLAY:
        r_ncf = all_results[f"{name}_ncf"]
        r_cf  = all_results[f"{name}_cf"]
        mcnemar_results[name] = mcnemar_test(
            r_ncf["y_true"], r_ncf["y_pred"], r_cf["y_pred"])
        delong_results[name]  = delong_auc_test(
            r_ncf["y_true"], r_ncf["y_prob"], r_cf["y_prob"])

    print_summary(all_results, mcnemar_results, delong_results)

    print("\n>> STEP 5  Generating Plots")
    plot_roc_curves(all_results, PLOTS_DIR)
    plot_pr_curves(all_results, PLOTS_DIR)
    plot_calibration(all_results, PLOTS_DIR)
    plot_confusion_matrices(all_results, PLOTS_DIR)
    plot_metrics_comparison(all_results, PLOTS_DIR)
    plot_fpr_fnr_delta(all_results, PLOTS_DIR)
    plot_summary_heatmap(all_results, PLOTS_DIR)

    print("\n>> STEP 6  Exporting Results to JSON")
    export_json(all_results, mcnemar_results, delong_results, OUT_DIR)

    print(f"\n{'='*70}")
    print(f"  Pipeline complete!")
    print(f"  Results -> {OUT_DIR}")
    print(f"  Plots   -> {PLOTS_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
