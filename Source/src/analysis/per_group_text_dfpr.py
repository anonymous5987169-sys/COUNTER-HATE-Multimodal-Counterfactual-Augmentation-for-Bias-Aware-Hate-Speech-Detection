"""
per_group_text_dfpr.py
=======================
Comprehensive per-identity-group DFPR (Differential False Positive Rate)
analysis for ALL text models: nCF (non-counterfactual) vs CF (counterfactual).

Computes FPR, FNR per identity group for every saved text model, generates
heatmaps, bar charts, and a summary JSON.

Outputs
───────
  analysis/results/text_per_group_dfpr_results.json
  analysis/results/plots/text_per_group_fpr_heatmap.png
  analysis/results/plots/text_dfpr_by_group.png
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR  = ANALYSIS_DIR / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"
for _d in (RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Prediction CSVs (MiniLM+MLP only — other models need re-prediction)
PRED_DIR = PROJECT_ROOT / "text_models" / "enhanced_results" / "predictions"

# Saved model directory
MODELS_DIR = PROJECT_ROOT / "text_models" / "enhanced_results" / "models"

# Cached MiniLM embeddings
EMBED_DIR = PROJECT_ROOT / "text_models" / "enhanced_results" / "embeddings"

# Dataset
DATA_18K = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k.csv"
DATA_6K  = PROJECT_ROOT / "src" / "counterfactual_gen" / "hate_speech_dataset_6k.csv"

# Canonical splits
sys.path.insert(0, str(PROJECT_ROOT))
from canonical_splits import get_canonical_splits  # noqa: E402

# ─── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# ─── Model registry ──────────────────────────────────────────────────────────
# Maps model short name -> (feature_type, ncf_joblib_name, cf_joblib_name)
# feature_type: "minilm" | "tfidf_word" | "tfidf_enhanced"
MODEL_REGISTRY = {
    "MiniLM+MLP": ("minilm", "minilm_mlp_ncf.joblib", "minilm_mlp_cf.joblib"),
    "MiniLM+LR":  ("minilm", "minilm_lr_ncf.joblib",  "minilm_lr_cf.joblib"),
    "MiniLM+SVM": ("minilm", "minilm_svm_ncf.joblib",  "minilm_svm_cf.joblib"),
    "LR/TF-IDF":  ("tfidf_word", "lr_tfidf_ncf.joblib",  "lr_tfidf_cf.joblib"),
    "Ridge/TF-IDF": ("tfidf_word", "ridge_tfidf_ncf.joblib", "ridge_tfidf_cf.joblib"),
    "NB/TF-IDF":  ("tfidf_word", "nb_tfidf_ncf.joblib",  "nb_tfidf_cf.joblib"),
    "RF/TF-IDF":  ("tfidf_word", "rf_tfidf_ncf.joblib",  "rf_tfidf_cf.joblib"),
    "SVM/TF-IDF": ("tfidf_word", "svm_tfidf_ncf.joblib", "svm_tfidf_cf.joblib"),
    "SVM/TF-IDF+Char": ("tfidf_enhanced", "svm_enhanced_tfidf_ncf.joblib", "svm_enhanced_tfidf_cf.joblib"),
    "LR/TF-IDF+Char":  ("tfidf_enhanced", "lr_enhanced_tfidf_ncf.joblib",  "lr_enhanced_tfidf_cf.joblib"),
}

# Display order for plots
MODEL_ORDER = [
    "LR/TF-IDF", "Ridge/TF-IDF", "NB/TF-IDF", "RF/TF-IDF", "SVM/TF-IDF",
    "SVM/TF-IDF+Char", "LR/TF-IDF+Char",
    "MiniLM+LR", "MiniLM+SVM", "MiniLM+MLP",
]

# Canonical group order
GROUP_ORDER = [
    "race/ethnicity", "religion", "gender", "sexual_orientation",
    "national_origin/citizenship", "disability", "age", "other", "multiple/none",
]


# ═════════════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING — build test set with group labels
# ═════════════════════════════════════════════════════════════════════════════

def _binary_label(polarity):
    return 1 if str(polarity).strip().lower() == "hate" else 0


def _derive_group_label(class_label: str, target_group: str) -> str:
    """Map a sample to its identity group (same logic as enhanced_analysis.py)."""
    _CLASS_TO_GROUP = {
        "hate_race":     "race/ethnicity",
        "hate_religion": "religion",
        "hate_gender":   "gender",
        "hate_other":    "other",
    }
    return _CLASS_TO_GROUP.get(class_label, target_group)


def _filter_non_english(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["text"].fillna("").apply(
        lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1) > 0.05
    )
    if mask.sum():
        print(f"  Dropped {mask.sum()} non-English rows.")
    return df[~mask].copy()


def load_test_set() -> pd.DataFrame:
    """Load the canonical test split with group labels and binary labels."""
    print("[1] Loading test set …")
    df = pd.read_csv(DATA_18K)
    df = _filter_non_english(df)
    df["binary_label"] = df["polarity"].apply(_binary_label)

    originals = df[df["cf_type"] == "original"].copy()
    canon = get_canonical_splits()
    test_df = originals[originals["original_sample_id"].isin(canon["test_ids"])].copy()
    test_df = test_df.sort_values("original_sample_id").reset_index(drop=True)

    test_df["group_label"] = [
        _derive_group_label(cl, tg)
        for cl, tg in zip(test_df["class_label"].values, test_df["target_group"].values)
    ]

    print(f"  Test set: {len(test_df)} samples, "
          f"{test_df['binary_label'].mean()*100:.1f}% hate")
    print(f"  Groups: {dict(test_df['group_label'].value_counts())}")
    return test_df


# ═════════════════════════════════════════════════════════════════════════════
#  2.  FEATURE EXTRACTION for test set
# ═════════════════════════════════════════════════════════════════════════════

def get_minilm_test_features() -> np.ndarray:
    """Load cached MiniLM test embeddings."""
    emb_path = EMBED_DIR / "minilm_test_892.npy"
    if emb_path.exists():
        X = np.load(emb_path)
        print(f"  MiniLM test embeddings loaded: {X.shape}")
        return X
    raise FileNotFoundError(f"MiniLM test embeddings not found at {emb_path}")


def get_tfidf_word_test_features(test_df: pd.DataFrame) -> np.ndarray:
    """Transform test texts using saved TF-IDF word vectorizer."""
    vec_path = MODELS_DIR / "tfidf_word_ncf.joblib"
    if not vec_path.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {vec_path}")
    vectorizer = joblib.load(vec_path)
    X = vectorizer.transform(test_df["text"].fillna(""))
    print(f"  TF-IDF word features: {X.shape}")
    return X


def get_tfidf_enhanced_test_features(test_df: pd.DataFrame) -> np.ndarray:
    """Rebuild enhanced TF-IDF (word+char) test features.

    Since the enhanced vectorizers were not saved separately, we rebuild
    them from the nCF training data. This is consistent because vectorizers
    are fit on training data only.
    """
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("  Building enhanced TF-IDF vectorizers from nCF training data …")

    # Load nCF training data
    if DATA_6K.exists():
        df_6k = pd.read_csv(DATA_6K)
        df_6k = _filter_non_english(df_6k)
        df_6k = df_6k.rename(columns={"sample_id": "original_sample_id"})
    else:
        df_full = pd.read_csv(DATA_18K)
        df_full = _filter_non_english(df_full)
        df_6k = df_full[df_full["cf_type"] == "original"].copy()

    canon = get_canonical_splits()
    train_texts = df_6k[df_6k["original_sample_id"].isin(canon["train_ids"])]["text"].fillna("")

    vec_word = TfidfVectorizer(
        max_features=12_000, ngram_range=(1, 3),
        sublinear_tf=True, analyzer="word",
        strip_accents="unicode", min_df=2, stop_words="english",
    )
    vec_char = TfidfVectorizer(
        max_features=8_000, ngram_range=(2, 4),
        sublinear_tf=True, analyzer="char_wb",
        strip_accents="unicode", min_df=3,
    )
    vec_word.fit(train_texts)
    vec_char.fit(train_texts)

    test_texts = test_df["text"].fillna("")
    X = sparse.hstack([vec_word.transform(test_texts),
                        vec_char.transform(test_texts)], format="csr")
    print(f"  Enhanced TF-IDF features: {X.shape}")
    return X


# ═════════════════════════════════════════════════════════════════════════════
#  3.  PER-GROUP FPR / FNR COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_group_rates(y_true: np.ndarray, y_pred: np.ndarray,
                        group_labels: np.ndarray) -> dict:
    """Compute FPR and FNR per identity group.

    Returns dict {group: {fpr, fnr, n_total, n_neg, n_pos}}.
    """
    results = {}
    for g in sorted(set(group_labels)):
        mask = (group_labels == g)
        yt = y_true[mask]
        yp = y_pred[mask]

        neg_mask = (yt == 0)
        pos_mask = (yt == 1)
        n_neg = int(neg_mask.sum())
        n_pos = int(pos_mask.sum())

        fpr = float((yp[neg_mask] == 1).sum() / n_neg) if n_neg > 0 else None
        fnr = float((yp[pos_mask] == 0).sum() / n_pos) if n_pos > 0 else None

        results[g] = {
            "fpr": fpr,
            "fnr": fnr,
            "n_total": int(mask.sum()),
            "n_neg": n_neg,
            "n_pos": n_pos,
        }
    return results


def compute_dfpr_all_models(test_df: pd.DataFrame) -> dict:
    """Compute per-group FPR/FNR and DFPR for all saved text models.

    Returns nested dict: {model_name: {group: {fpr_ncf, fpr_cf, dfpr, ...}}}
    """
    y_true = test_df["binary_label"].values
    group_labels = test_df["group_label"].values

    # Pre-compute features for each feature type
    feature_cache = {}

    all_results = {}
    skipped = []

    for model_name in MODEL_ORDER:
        if model_name not in MODEL_REGISTRY:
            continue

        feat_type, ncf_file, cf_file = MODEL_REGISTRY[model_name]
        ncf_path = MODELS_DIR / ncf_file
        cf_path  = MODELS_DIR / cf_file

        # Check model files exist
        if not ncf_path.exists() or not cf_path.exists():
            print(f"  SKIP {model_name}: model file(s) missing")
            skipped.append(model_name)
            continue

        # Get features (cached by type)
        if feat_type not in feature_cache:
            try:
                if feat_type == "minilm":
                    feature_cache[feat_type] = get_minilm_test_features()
                elif feat_type == "tfidf_word":
                    feature_cache[feat_type] = get_tfidf_word_test_features(test_df)
                elif feat_type == "tfidf_enhanced":
                    feature_cache[feat_type] = get_tfidf_enhanced_test_features(test_df)
            except Exception as e:
                print(f"  SKIP {model_name}: feature extraction failed: {e}")
                skipped.append(model_name)
                continue

        if feat_type not in feature_cache:
            skipped.append(model_name)
            continue

        X_test = feature_cache[feat_type]

        # Load models and predict
        print(f"  Predicting: {model_name} …")
        try:
            model_ncf = joblib.load(ncf_path)
            model_cf  = joblib.load(cf_path)

            y_pred_ncf = model_ncf.predict(X_test)
            y_pred_cf  = model_cf.predict(X_test)
        except Exception as e:
            print(f"  SKIP {model_name}: prediction failed: {e}")
            skipped.append(model_name)
            continue

        # Compute per-group rates
        rates_ncf = compute_group_rates(y_true, y_pred_ncf, group_labels)
        rates_cf  = compute_group_rates(y_true, y_pred_cf,  group_labels)

        model_result = {}
        for g in GROUP_ORDER:
            if g not in rates_ncf and g not in rates_cf:
                continue
            r_ncf = rates_ncf.get(g, {})
            r_cf  = rates_cf.get(g, {})

            fpr_ncf = r_ncf.get("fpr")
            fpr_cf  = r_cf.get("fpr")
            fnr_ncf = r_ncf.get("fnr")
            fnr_cf  = r_cf.get("fnr")

            dfpr = (fpr_cf - fpr_ncf) if (fpr_ncf is not None and fpr_cf is not None) else None
            dfnr = (fnr_cf - fnr_ncf) if (fnr_ncf is not None and fnr_cf is not None) else None

            model_result[g] = {
                "fpr_ncf": fpr_ncf,
                "fpr_cf":  fpr_cf,
                "dfpr":    dfpr,
                "fnr_ncf": fnr_ncf,
                "fnr_cf":  fnr_cf,
                "dfnr":    dfnr,
                "count":   r_ncf.get("n_total", r_cf.get("n_total", 0)),
                "n_neg":   r_ncf.get("n_neg", 0),
                "n_pos":   r_ncf.get("n_pos", 0),
            }

        all_results[model_name] = model_result

    # ── Also compute from saved prediction CSVs for MiniLM+MLP (cross-check)
    _add_from_prediction_csvs(all_results, "MiniLM+MLP (CSV)")

    if skipped:
        print(f"\n  ⚠ Skipped models: {skipped}")

    return all_results


def _add_from_prediction_csvs(all_results: dict, label: str):
    """Add MiniLM+MLP results directly from saved prediction CSVs (cross-check)."""
    cf_csv  = PRED_DIR / "minilm_mlp_cf_predictions.csv"
    ncf_csv = PRED_DIR / "minilm_mlp_ncf_predictions.csv"

    if not cf_csv.exists() or not ncf_csv.exists():
        print(f"  Prediction CSVs not found — skipping CSV cross-check.")
        return

    df_cf  = pd.read_csv(cf_csv)
    df_ncf = pd.read_csv(ncf_csv)

    model_result = {}
    for g in GROUP_ORDER:
        mask_cf  = (df_cf["group_label"] == g)
        mask_ncf = (df_ncf["group_label"] == g)

        if not mask_cf.any() and not mask_ncf.any():
            continue

        sub_cf  = df_cf[mask_cf]
        sub_ncf = df_ncf[mask_ncf]

        # FPR: among true negatives (true_label==0), fraction predicted 1
        neg_cf  = sub_cf[sub_cf["true_label"] == 0]
        neg_ncf = sub_ncf[sub_ncf["true_label"] == 0]

        fpr_cf  = float((neg_cf["pred_label"] == 1).sum() / len(neg_cf)) if len(neg_cf) > 0 else None
        fpr_ncf = float((neg_ncf["pred_label"] == 1).sum() / len(neg_ncf)) if len(neg_ncf) > 0 else None

        # FNR: among true positives (true_label==1), fraction predicted 0
        pos_cf  = sub_cf[sub_cf["true_label"] == 1]
        pos_ncf = sub_ncf[sub_ncf["true_label"] == 1]

        fnr_cf  = float((pos_cf["pred_label"] == 0).sum() / len(pos_cf)) if len(pos_cf) > 0 else None
        fnr_ncf = float((pos_ncf["pred_label"] == 0).sum() / len(pos_ncf)) if len(pos_ncf) > 0 else None

        dfpr = (fpr_cf - fpr_ncf) if (fpr_ncf is not None and fpr_cf is not None) else None
        dfnr = (fnr_cf - fnr_ncf) if (fnr_ncf is not None and fnr_cf is not None) else None

        model_result[g] = {
            "fpr_ncf": fpr_ncf,
            "fpr_cf":  fpr_cf,
            "dfpr":    dfpr,
            "fnr_ncf": fnr_ncf,
            "fnr_cf":  fnr_cf,
            "dfnr":    dfnr,
            "count":   int(max(mask_cf.sum(), mask_ncf.sum())),
            "n_neg":   int(max(len(neg_cf), len(neg_ncf))),
            "n_pos":   int(max(len(pos_cf), len(pos_ncf))),
        }

    all_results[label] = model_result


# ═════════════════════════════════════════════════════════════════════════════
#  4.  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_fpr_heatmap(all_results: dict, save_path: Path):
    """Per-group FPR heatmap: models (rows) × identity groups (columns).

    Two side-by-side panels: nCF and CF.
    """
    models_to_plot = [m for m in MODEL_ORDER if m in all_results]
    groups = [g for g in GROUP_ORDER if any(g in all_results[m] for m in models_to_plot)]

    # Build matrices
    fpr_ncf_mat = np.full((len(models_to_plot), len(groups)), np.nan)
    fpr_cf_mat  = np.full((len(models_to_plot), len(groups)), np.nan)

    for i, m in enumerate(models_to_plot):
        for j, g in enumerate(groups):
            if g in all_results[m]:
                v_ncf = all_results[m][g].get("fpr_ncf")
                v_cf  = all_results[m][g].get("fpr_cf")
                if v_ncf is not None:
                    fpr_ncf_mat[i, j] = v_ncf
                if v_cf is not None:
                    fpr_cf_mat[i, j] = v_cf

    # Short group labels for readability
    short_groups = [g.replace("/ethnicity", "").replace("national_origin/citizenship", "nat_origin")
                      .replace("sexual_orientation", "sex_orient")
                      .replace("multiple/none", "multi/none")
                    for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, mat, title in [
        (axes[0], fpr_ncf_mat, "FPR — nCF (no augmentation)"),
        (axes[1], fpr_cf_mat,  "FPR — CF (counterfactual augmented)"),
    ]:
        mask = np.isnan(mat)
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=short_groups, yticklabels=models_to_plot if ax == axes[0] else False,
            vmin=0, vmax=1, mask=mask,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "False Positive Rate"},
        )
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Identity Group")
        if ax == axes[0]:
            ax.set_ylabel("Model")

    fig.suptitle("Per-Group False Positive Rate: All Text Models",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {save_path}")


def plot_dfpr_bars(all_results: dict, save_path: Path):
    """DFPR bar chart: grouped bars (one per model) for each identity group."""
    models_to_plot = [m for m in MODEL_ORDER if m in all_results]
    groups = [g for g in GROUP_ORDER if any(g in all_results[m] for m in models_to_plot)]

    # Build data for plotting
    rows = []
    for m in models_to_plot:
        for g in groups:
            if g in all_results[m]:
                dfpr = all_results[m][g].get("dfpr")
                if dfpr is not None:
                    rows.append({"Model": m, "Group": g, "DFPR": dfpr})

    if not rows:
        print("  No DFPR data to plot.")
        return

    plot_df = pd.DataFrame(rows)

    # Short group labels
    group_short = {
        "race/ethnicity": "race",
        "religion": "religion",
        "gender": "gender",
        "sexual_orientation": "sex_orient",
        "national_origin/citizenship": "nat_origin",
        "disability": "disability",
        "age": "age",
        "other": "other",
        "multiple/none": "multi/none",
    }
    plot_df["Group_short"] = plot_df["Group"].map(group_short)

    n_groups = len(groups)
    n_models = len(models_to_plot)

    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2), 7))

    # Use a colour palette
    palette = sns.color_palette("tab10", n_colors=n_models)

    bar_width = 0.8 / n_models
    x = np.arange(n_groups)

    for i, m in enumerate(models_to_plot):
        vals = []
        for g in groups:
            if g in all_results[m] and all_results[m][g].get("dfpr") is not None:
                vals.append(all_results[m][g]["dfpr"])
            else:
                vals.append(0.0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width, label=m,
                      color=palette[i], edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([group_short.get(g, g) for g in groups], rotation=30, ha="right")
    ax.set_ylabel("ΔFPR  (CF − nCF)", fontsize=11)
    ax.set_xlabel("Identity Group", fontsize=11)
    ax.set_title("Differential FPR by Identity Group — All Text Models",
                 fontweight="bold", fontsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)

    # Add zero reference note
    ax.annotate("← CF reduces FPR     CF increases FPR →",
                xy=(0.5, 0.02), xycoords="axes fraction",
                ha="center", fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  DFPR bar chart saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  5.  SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(all_results: dict):
    """Print a formatted summary table to stdout."""
    models_to_show = [m for m in MODEL_ORDER if m in all_results]
    # Also include CSV cross-check if present
    if "MiniLM+MLP (CSV)" in all_results:
        models_to_show.append("MiniLM+MLP (CSV)")

    groups = [g for g in GROUP_ORDER if any(g in all_results.get(m, {}) for m in models_to_show)]

    # Short group names
    short = {
        "race/ethnicity": "race", "religion": "relig", "gender": "gender",
        "sexual_orientation": "s_orient", "national_origin/citizenship": "nat_orig",
        "disability": "disab", "age": "age", "other": "other",
        "multiple/none": "multi",
    }

    print("\n" + "=" * 120)
    print("PER-GROUP TEXT MODEL DFPR ANALYSIS")
    print("=" * 120)

    # ── FPR nCF table ──
    print("\n┌─ FPR (nCF) — False Positive Rate under non-counterfactual models")
    header = f"{'Model':<22}" + "".join(f"{short.get(g,g):>10}" for g in groups)
    print(header)
    print("-" * len(header))
    for m in models_to_show:
        row = f"{m:<22}"
        for g in groups:
            v = all_results.get(m, {}).get(g, {}).get("fpr_ncf")
            row += f"{v:10.4f}" if v is not None else f"{'N/A':>10}"
        print(row)

    # ── FPR CF table ──
    print("\n┌─ FPR (CF) — False Positive Rate under counterfactual-augmented models")
    print(header)
    print("-" * len(header))
    for m in models_to_show:
        row = f"{m:<22}"
        for g in groups:
            v = all_results.get(m, {}).get(g, {}).get("fpr_cf")
            row += f"{v:10.4f}" if v is not None else f"{'N/A':>10}"
        print(row)

    # ── DFPR table ──
    print("\n┌─ ΔFPR (CF − nCF) — positive = CF increased FPR, negative = CF reduced FPR")
    print(header)
    print("-" * len(header))
    for m in models_to_show:
        row = f"{m:<22}"
        for g in groups:
            v = all_results.get(m, {}).get(g, {}).get("dfpr")
            if v is not None:
                sign = "+" if v > 0 else ""
                row += f"{sign}{v:9.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    # ── Group counts ──
    print(f"\n┌─ Group sample counts (test set)")
    row = f"{'Count':<22}"
    # Use first available model for counts
    ref_model = models_to_show[0] if models_to_show else None
    if ref_model:
        for g in groups:
            c = all_results.get(ref_model, {}).get(g, {}).get("count", 0)
            row += f"{c:>10}"
        print(row)

    print("=" * 120)


# ═════════════════════════════════════════════════════════════════════════════
#  6.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "━" * 70)
    print("  Per-Group Text DFPR Analysis")
    print("━" * 70)

    # Step 1: Load test set
    test_df = load_test_set()

    # Step 2: Compute per-group FPR/FNR for all models
    print("\n[2] Computing per-group FPR/FNR for all text models …")
    all_results = compute_dfpr_all_models(test_df)

    # Step 3: Print summary
    print_summary_table(all_results)

    # Step 4: Plots
    print("\n[3] Generating plots …")
    plot_fpr_heatmap(all_results, PLOTS_DIR / "text_per_group_fpr_heatmap.png")
    plot_dfpr_bars(all_results, PLOTS_DIR / "text_dfpr_by_group.png")

    # Step 5: Save JSON
    # Convert any numpy types for JSON serialisation
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_results = {}
    for model_name, model_data in all_results.items():
        json_results[model_name] = {}
        for group, group_data in model_data.items():
            json_results[model_name][group] = {
                k: _json_safe(v) for k, v in group_data.items()
            }

    # Add metadata
    output = {
        "metadata": {
            "description": "Per-identity-group DFPR analysis for all text models (nCF vs CF)",
            "test_set_size": len(test_df),
            "groups": GROUP_ORDER,
            "models_evaluated": list(all_results.keys()),
            "metric": "DFPR = FPR_CF - FPR_nCF",
        },
        "per_model_per_group": json_results,
    }

    json_path = RESULTS_DIR / "text_per_group_dfpr_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results JSON saved → {json_path}")

    # Final summary
    n_models = len(all_results)
    n_groups = len(set(g for m in all_results.values() for g in m))
    print(f"\n✓ Analysis complete: {n_models} models × {n_groups} groups")
    print(f"  Files created:")
    print(f"    {PLOTS_DIR / 'text_per_group_fpr_heatmap.png'}")
    print(f"    {PLOTS_DIR / 'text_dfpr_by_group.png'}")
    print(f"    {json_path}")


if __name__ == "__main__":
    main()
