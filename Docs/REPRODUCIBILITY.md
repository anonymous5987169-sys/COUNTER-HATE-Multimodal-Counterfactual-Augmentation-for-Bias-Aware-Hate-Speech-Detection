# Reproducibility Guide — Bias Evaluation of CAD in Hate Speech Detection

**Target venue:** ACM Multimedia 2026  
**Last updated:** March 2026

This document provides step-by-step instructions to reproduce every result cited in [`prof-report.md`](prof-report.md). It is written for human reviewers, collaborators, and AI agents alike.

## Post-Image-Rerun Addendum (2026-03-20)

This repository was re-run after image replacement using CPU-only execution on `pyenv 3.12.0`, with baseline pipeline excluded and DistilBERT not executed.

Executed order:
1. `python image_models/run_all.py`
2. `python cross_modal/late_fusion_ensemble.py`
3. `python cross_modal/ablation_calibration_study.py`
4. `python cross_modal/run_all.py`
5. `python cross_modal/cross_attention_fusion.py` (standalone rerun after orchestration fix)
6. `python analysis/run_all.py --skip-text --skip-image --skip-cross-modal`
7. `python scripts/generate_all_plots.py`
8. `pytest tests/ -v --tb=short -x`

Fusion condition coverage now includes all three image conditions in consolidated outputs:
- `nCF`
- `CF-no-adv`
- `CF+GRL`

Refreshed canonical image metrics (`image_models/results/evaluation_results.json`):
- `nCF`: F1 `0.7809`, AUC `0.8322`, FPR `0.4009`
- `CF-no-adv`: F1 `0.8080`, AUC `0.8474`, FPR `0.3333`
- `CF+GRL`: F1 `0.7885`, AUC `0.8401`, FPR `0.3649`

Refreshed fusion summaries:
- Late fusion equal-weight F1: `0.8645` (`nCF`), `0.8526` (`CF-no-adv`), `0.8520` (`CF+GRL`)
- Cross-attention ensemble F1: `0.8604` (`nCF`), `0.8300` (`CF-no-adv`), `0.8378` (`CF+GRL`)

Validation status:
- Step-4 sanity gate passed: required fusion JSON files exist, include all 3 conditions, and contain no NaN/Inf values.
- CF-image linkage validation passed (no missing CF image mappings in the rerun check).
- `pytest` currently passes (`12 passed`).

Deferred/non-blocking items:
- Multi-seed rerun remains deferred in this cycle.
- `analysis/run_all.py --skip-text --skip-image --skip-cross-modal` now completes with all listed steps passing (including `GRL Hyperparameter Ablation`).

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Dataset & Canonical Splits](#2-dataset--canonical-splits)
3. [Pipeline Execution Order (Dependency Graph)](#3-pipeline-execution-order-dependency-graph)
4. [Pipeline 1 — Text Models](#4-pipeline-1--text-models)
5. [Pipeline 2 — Image Models](#5-pipeline-2--image-models)
6. [Pipeline 3 — Cross-Modal Fusion](#6-pipeline-3--cross-modal-fusion)
7. [Pipeline 4 — Statistical Analysis](#7-pipeline-4--statistical-analysis)
8. [Pipeline 5 — Multi-Seed Robustness](#8-pipeline-5--multi-seed-robustness)
9. [Pipeline 6 — Plot Generation & Comprehensive Evaluation](#9-pipeline-6--plot-generation--comprehensive-evaluation)
10. [Pipeline 7 — Baseline Pipeline (Optional, CPU-Only)](#10-pipeline-7--baseline-pipeline-optional-cpu-only)
11. [Running via Docker](#11-running-via-docker)
12. [Smoke-Test Mode (Quick Validation)](#12-smoke-test-mode-quick-validation)
13. [Master Result File Reference](#13-master-result-file-reference)
14. [JSON Output Schemas](#14-json-output-schemas)
15. [Result Extraction Guide](#15-result-extraction-guide)
16. [Troubleshooting](#16-troubleshooting)
17. [Known Caveats & Limitations](#17-known-caveats--limitations)

---

## 1. Prerequisites & Environment Setup

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 16 GB | 32 GB |
| GPU | Not required (CPU fallback) | NVIDIA GPU with ≥4 GB VRAM (CUDA) |
| Disk | 20 GB free | 50 GB free (images + models) |

> **Note:** The image pipeline and cross-attention fusion benefit from GPU acceleration but will run on CPU. The baseline pipeline is CPU-only by design.

### 1.2 Software Requirements

- **Python 3.10+** (main project) — tested on 3.10, 3.14
- **Python 3.12** (baseline pipeline only, via pyenv — optional)
- **Git**, **pip**, standard POSIX tools

### 1.3 Main Environment Setup

```bash
cd /path/to/major-project

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Verify critical packages
python3 -c "import torch; import sentence_transformers; import sklearn; print('OK')"
```

**Key packages installed by `requirements.txt`:**

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.2.0 | Neural network training (image, fusion) |
| `torchvision` | ≥0.17.0 | EfficientNet-B0 backbone, image transforms |
| `sentence-transformers` | ≥2.6.1 | MiniLM-L12-v2 text embeddings |
| `transformers` | ≥4.39.3 | HuggingFace model hub |
| `scikit-learn` | ≥1.4.1 | Classical ML classifiers, metrics |
| `scipy` | ≥1.12.0 | Statistical tests (Wilcoxon, Chi-square) |
| `statsmodels` | ≥0.14.1 | OLS regression, ANOVA |
| `pandas` | ≥2.2.1 | Data manipulation |
| `numpy` | ≥1.26.4 | Numerical operations |
| `matplotlib` | ≥3.8.3 | Plot generation |
| `seaborn` | ≥0.13.2 | Statistical visualizations |

### 1.4 Environment Variable

The project root must be on `PYTHONPATH` for cross-module imports:

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

All `run_all.py` scripts set this automatically via `sys.path` insertion, but it is good practice to set it in your shell.

### 1.5 Baseline Pipeline Environment (Optional)

```bash
# Only needed if running baseline-pipeline/
cd baseline-pipeline
pyenv install 3.12.0        # if not already installed
pyenv shell 3.12.0
python3 -m venv .venv-baseline
source .venv-baseline/bin/activate
pip install -r requirements.txt
```

---

## 2. Dataset & Canonical Splits

### 2.1 Dataset File

| Item | Value |
|---|---|
| **Path** | `data/datasets/final_dataset_18k.csv` |
| **Rows** | ~18,000 (6K originals + 12K counterfactuals) |
| **Required columns** | `original_sample_id`, `counterfactual_id`, `text`, `class_label`, `target_group`, `polarity`, `hate_score`, `confidence`, `cf_type` |

### 2.2 Canonical Splits — Single Source of Truth

**File:** [`canonical_splits.py`](canonical_splits.py)  
**Persisted artifact:** `data/splits/canonical_splits.json`

All pipelines import splits from this one module. Never create ad-hoc splits.

| Split | Size (originals) | Ratio | Contents |
|---|---|---|---|
| Train | 4,158 | 69.99% (target 70%) | Originals + all counterfactuals (CF condition only) |
| Val | 891 | 15.00% (target 15%) | Originals only — used for threshold tuning |
| Test | 892 | 15.01% (target 15%) | Originals only — the "fusion_test_900" set |

**Splitting strategy:** Stratified on `class_label` (8-class), grouped by `original_sample_id` (each original and its counterfactuals appear in exactly one split). Target ratio is 70/15/15; realized ratio is 69.99/15.00/15.01 (4,158/891/892 over 5,941 post-filter originals). `random_state=42`.

For **CF condition threshold tuning**, validation is CF-augmented (originals + their counterfactual variants) while the test split remains originals-only.

`canonical_splits.json` metadata also records `n_originals_raw` and `n_removed_non_english` for traceability (e.g., 5,970 raw originals -> 5,941 post-filter after removing 29 non-English rows).

### 2.3 Validate Splits

```bash
python3 scripts/validate_canonical_splits.py
# Expected output: "VALID" with exit code 0
```

This checks: key schema, pairwise disjointness, count consistency, union coverage, class distribution, and closeness to the target 70/15/15 split.

### 2.4 Image Data

Generated images (~18K PNGs, 720×720) live in:

```
Hate/
  Hate_race/generated_images/
  Hate_religion/generated_images/
  Hate_Gender/generated_images/
  Hate_Others/generated_images/
non-hate/
  generated_images-offensive-non-hate/
  generated_images-neutral/
  generated_images-counter-speech/
  generated_images-ambigious/
```

These images were generated offline via Z-Image-Turbo T2I and must be present before running image or fusion pipelines.

---

## 3. Pipeline Execution Order (Dependency Graph)

```
                         ┌──────────────────────┐
                         │   0. Validate Splits  │
                         │   (optional check)    │
                         └──────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │  1. TEXT MODELS  │             │  2. IMAGE MODELS │
          │  (independent)  │             │  (independent)   │
          └────────┬────────┘             └────────┬─────────┘
                   │                               │
                   │   Requires BOTH text & image  │
                   │   predictions/models           │
                   └───────────────┬───────────────┘
                                   ▼
                      ┌────────────────────────┐
                      │  3. CROSS-MODAL FUSION │
                      │  (depends on 1 + 2)    │
                      └────────────┬───────────┘
                                   │
                   ┌───────────────┼──────────────────┐
                   ▼               ▼                  ▼
          ┌──────────────┐ ┌──────────────┐  ┌──────────────────┐
          │ 4. ANALYSIS  │ │ 5. MULTI-SEED│  │ 6. PLOTS &       │
          │ (depends: 3) │ │ (indep.)     │  │ COMPREHENSIVE    │
          └──────────────┘ └──────────────┘  │ EVAL (dep: 1+2+3)│
                                             └──────────────────┘
```

**Execution order (recommended):**

| Step | Pipeline | Depends on | Command |
|---|---|---|---|
| 0 | Validate splits | — | `python3 scripts/validate_canonical_splits.py` |
| 1 | Text models | Dataset | `python3 text_models/run_all.py` |
| 2 | Image models | Dataset + images | `python3 image_models/run_all.py` |
| 3 | Cross-modal fusion | Steps 1 + 2 | `python3 cross_modal/run_all.py` |
| 4 | Statistical analysis | Steps 1 + 2 + 3 | `python3 analysis/run_all.py --skip-text --skip-image --skip-cross-modal` |
| 5 | Multi-seed robustness | Dataset + images | `python3 scripts/multi_seed_experiment.py` |
| 6 | Plots + comprehensive eval | Steps 1 + 2 + 3 | `python3 scripts/generate_all_plots.py` |

> **Steps 1 and 2 are independent** — they can run in parallel on separate machines or GPUs.  
> **Step 5 is independent** — it retrains from scratch with multiple seeds and does not depend on prior checkpoints.

---

## 4. Pipeline 1 — Text Models

### 4.1 What It Does

Trains text classifiers under two conditions (nCF = 6K, CF = 18K) and evaluates on the canonical 900-sample test set.

### 4.2 Commands

```bash
# Full run (produces all text model results)
python3 text_models/run_all.py

# Individual scripts (if you need to regenerate specific outputs):
python3 text_models/binary_fairness_analysis.py    # TF-IDF baselines only
python3 text_models/enhanced_analysis.py           # TF-IDF + MiniLM (8 models × 2 conditions)
```

### 4.3 Models Trained

| Feature Extractor | Classifiers | Conditions |
|---|---|---|
| TF-IDF (10K features, n-gram 1–2) | LogReg, Ridge, Naive Bayes, RF, SVM | nCF, CF |
| MiniLM-L12-v2 (384-dim) | LogReg, SVM, MLP(256,128) | nCF, CF |

### 4.4 Output Files

| File | Path | Description |
|---|---|---|
| Binary TF-IDF results | `text_models/binary_fairness_results/binary_fairness_results.json` | 4 TF-IDF models × 2 conditions with bootstrap CIs |
| Enhanced results | `text_models/enhanced_results/enhanced_results.json` | 8 models × 2 conditions — **⚠️ MiniLM CF rows are misaligned** |
| Trained models | `text_models/models/*.joblib` | Serialized sklearn/MLP classifiers |
| Predictions | `text_models/results/predictions/*.csv` | Per-sample predictions |
| Plots | `text_models/results/plots/` or `text_models/plots/` | ROC curves, confusion matrices |

### 4.5 Critical Caveat

> **Do NOT use `enhanced_results.json` for MiniLM CF metrics.** The MiniLM CF rows in that file were evaluated on a misaligned test set and report artificially low F1 (~0.56–0.63). The canonical MiniLM numbers come from `cross_modal/results/comprehensive_evaluation.json`, which uses the correctly aligned `fusion_test_900`.

### 4.6 Expected Key Results (Text)

| Model | Condition | F1 | AUC | FPR |
|---|---|---|---|---|
| TF-IDF + SVM | nCF | 0.828 | 0.888 | 0.212 |
| TF-IDF + SVM | CF | 0.810 | 0.877 | 0.266 |
| MiniLM + MLP | nCF | 0.863 | 0.919 | 0.237 |
| **MiniLM + MLP** | **CF** | **0.956** | **0.979** | **0.059** |

---

## 5. Pipeline 2 — Image Models

### 5.1 What It Does

Trains EfficientNet-B0 on generated images under 3 conditions and evaluates per-group fairness.

### 5.2 Commands

```bash
# Full run (train + evaluate all 3 conditions)
python3 image_models/run_all.py

# GPU-accelerated
python3 image_models/run_all.py --device cuda

# Evaluate only (skip training, load saved checkpoints)
python3 image_models/run_all.py --eval-only
```

### 5.3 CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | config default | Training epochs |
| `--batch-size` | config default | Batch size |
| `--lr-backbone` | config default | Backbone learning rate |
| `--lr-heads` | config default | Classification head learning rate |
| `--patience` | config default | Early stopping patience |
| `--adv-weight` | `0.5` | GRL adversarial loss weight |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--smoke-test` | off | 100 samples, 2 epochs, skip bootstrap |
| `--eval-only` | off | Load checkpoints, evaluate only |

### 5.4 Conditions Trained

| Condition | Training Data | Adversarial Head |
|---|---|---|
| `nCF` | 6K originals | No |
| `CF-no-adv` | 18K (originals + counterfactuals) | No |
| `CF+GRL` | 18K (originals + counterfactuals) | Yes (Gradient Reversal Layer) |

### 5.5 Output Files

| File | Path | Description |
|---|---|---|
| **Canonical image results** | `image_models/results/evaluation_results.json` | F1, AUC, FPR, per-group FPR, EO-diff, bootstrap CIs |
| Training log | `image_models/results/training_log.txt` | Per-epoch loss/accuracy |
| Checkpoints | `image_models/checkpoints/*.pth` | Saved model weights |
| Predictions | `image_models/results/predictions/*.csv` | Per-sample predictions |
| Plots | `image_models/results/plots/*.png` | ROC, confusion matrices |

### 5.6 Expected Key Results (Image)

| Condition | F1 | AUC | Overall FPR | Max Group FPR | EO-diff |
|---|---|---|---|---|---|
| nCF | 0.781 | 0.832 | 0.401 | 0.857 (Disability) | 0.680 |
| CF-no-adv | 0.808 | 0.847 | 0.333 | 0.600 (Age) | 0.670 |
| CF+GRL | 0.788 | 0.840 | 0.365 | 0.600 (Age) | 0.633 |

---

## 6. Pipeline 3 — Cross-Modal Fusion

### 6.1 What It Does

Combines the strongest text model (MiniLM+MLP CF) and image model (EfficientNet CF-no-adv) through 5 fusion strategies.

### 6.2 Prerequisites

**Before running this pipeline, you MUST have completed:**
- Pipeline 1 (Text Models) — needs trained MiniLM+MLP model and predictions
- Pipeline 2 (Image Models) — needs trained EfficientNet model and predictions

Specifically, the following artifacts must exist:
- `text_models/models/minilm_mlp_cf.joblib` (or equivalent saved model)
- `image_models/checkpoints/efficientnet_cf_no_adv.pth` (or equivalent)
- Text prediction CSVs in `text_models/results/predictions/`
- Image prediction CSVs in `image_models/results/predictions/`
- `cross_modal/results/predictions/fusion_train_oof_predictions.csv` and `cross_modal/results/predictions/fusion_test_predictions.csv` for leakage-safe stacking

### 6.3 Commands

```bash
# Full cross-modal pipeline (runs all 5 sub-steps in sequence)
python3 cross_modal/run_all.py

# Individual sub-steps (if regenerating specific results):
python3 cross_modal/late_fusion_ensemble.py          # Late fusion (Table 6)
python3 cross_modal/ablation_calibration_study.py     # Weight ablation + calibration
python3 cross_modal/stacking_ensemble.py              # Stacking meta-learner
python3 cross_modal/learned_fusion.py                 # 6 learned strategies
python3 cross_modal/cross_attention_fusion.py         # Cross-attention GMU + GRL

# Smoke-test mode
python3 cross_modal/run_all.py --smoke-test
python3 cross_modal/cross_attention_fusion.py --smoke-test
```

### 6.4 Sub-Pipeline Execution Order

The `run_all.py` runs these in sequence via subprocess:

| Order | Script | Output JSON | Description |
|---|---|---|---|
| 1 | `consistency_analysis.py` | (console output) | Text vs image prediction agreement |
| 2 | `stacking_ensemble.py` | `stacking_ensemble_results.json` | 4 meta-learners, train-OOF fitting + single held-out test evaluation |
| 3 | `learned_fusion.py` | `learned_fusion_results.json` | 6 fusion strategies || 5 | `cross_attention_fusion.py` | `cross_attention_fusion_results.json` | GMU + GRL (most complex) |

> **Note:** `late_fusion_ensemble.py` and `ablation_calibration_study.py` are not called by `run_all.py` — run them separately.

### 6.5 Output Files

| File | Path | Description |
|---|---|---|
| Late fusion | `cross_modal/results/late_fusion_results.json` | Equal/Learned/| Ablation + calibration | `cross_modal/results/ablation_calibration_results.json` | Weight sweep × calibration grid |
| Stacking | `cross_modal/results/stacking_ensemble_results.json` | Train-only 5-fold OOF for fitting/tuning, then one-shot test evaluation |
| Learned fusion | `cross_modal/results/learned_fusion_results.json` | 6 strategies comparison || Cross-attention | `cross_modal/results/cross_attention_fusion_results.json` | GMU + GRL 5-fold CV |
| Predictions | `cross_modal/results/predictions/*.csv` | Fusion-level predictions |
| Saved models | `cross_modal/models/` | Fusion model checkpoints |
| Plots | `cross_modal/results/plots/*.png` | Strategy comparison figures |

### 6.6 Expected Key Results (Fusion)

| Strategy | F1 | AUC | ECE |
|---|---|---|---|
| Late Fusion (w=0.50) | 0.853 | 0.910 | 0.052 |
| Weighted Avg (w=0.445) | 0.854 | 0.913 | 0.074 |
| Stacking (Meta-LR) | 0.854 | 0.905 | 0.120 |
| Learned (Product) | 0.861 | 0.894 | 0.128 |
| Cross-Attention GMU | 0.830 | 0.913 | 0.049 |

### 6.7 Cross-Attention Architecture Sanity Check

`cross_modal/cross_attention_fusion.py` implements feature-level fusion with:

- Text embeddings: `text_dim = 384` (MiniLM)
- Image backbone features: `image_raw_dim = 1280` (EfficientNet-B0)
- Image projection: `Linear(1280 -> 384)`
- Cross-attention concat: `[text_attn; image_attn; gated_text; gated_image]`

Therefore, the fusion width is:

`concat_dim = 4 * 384 = 1536`

This is intentional and arithmetically consistent. The `1536` does **not** imply asymmetric projection; it comes from concatenating four 384-d representations.

Parameter sanity check (from model instantiation):

- `trainable_params = 2,761,481`
- `total_params = 2,761,481`

---

## 7. Pipeline 4 — Statistical Analysis

### 7.1 What It Does

Runs comprehensive statistical tests on the predictions produced by Pipelines 1–3 to assess significance of CAD's effect on per-group FPR.

### 7.2 Prerequisites

Completed Pipelines 1, 2, and 3 (text + image predictions and fusion results must exist).

### 7.3 Commands

```bash
# Run analysis only (skip re-training text/image/fusion):
python3 analysis/run_all.py --skip-text --skip-image --skip-cross-modal

# Or run individual analysis scripts:
python3 analysis/enhanced_statistical_tests.py    # Chi-sq, ANOVA, Kruskal-Wallis
python3 analysis/statistical_tests.py             # Wilcoxon signed-rank
python3 analysis/mlp_cross_validation.py          # MiniLM+MLP bootstrap + nCF CV + CF size-ablation
python3 analysis/calibration_analysis.py          # ECE analysis
python3 analysis/error_analysis.py                # Misclassification analysis
python3 analysis/confidence_intervals.py          # Bootstrap CIs
python3 analysis/per_group_text_dfpr.py           # Per-group ΔFPR for text
python3 analysis/intersectional_bias.py           # Intersectional bias analysis
python3 analysis/baseline_comparison.py           # Baseline comparison
python3 analysis/clip_score_audit.py --report-only  # CLIP score audit
```

### 7.4 Full Pipeline Mode

If you want `analysis/run_all.py` to orchestrate everything (text → image → fusion → analysis):

```bash
# Full end-to-end (runs Pipelines 1+2+3+4 sequentially)
python3 analysis/run_all.py

# Smoke-test (fast validation, < 3 min)
python3 analysis/run_all.py --smoke-test
```

### 7.5 Output Files

| File | Path | Description |
|---|---|---|
| Enhanced stats | `analysis/results/enhanced_statistical_tests.json` | OLS ANOVA, Chi-square, Kruskal-Wallis, Cohen's d, per-group FPR |
| Wilcoxon | `analysis/results/wilcoxon_results.json` | Paired Wilcoxon signed-rank (image GRL) |
| MLP validation | `analysis/results/mlp_cv_results.json` | Bootstrap CI, nCF 5-fold CV, learning curves, CF size-ablation (6K/9K/12K/18K) |
| Calibration | `analysis/results/calibration_results.json` | ECE analysis |
| Error analysis | `analysis/results/error_analysis.json` | Misclassification patterns |
| CLIP scores | `analysis/results/clip_score_summary.json` | T2I quality audit |
| Baseline comparison | `analysis/results/baseline_comparison.json` | vs baseline pipeline |
| Intersectional bias | `analysis/results/intersectional_bias_results.json` | Cross-group analysis |
| Per-group text ΔFPR | `analysis/results/text_per_group_dfpr_results.json` | Text model per-group FPR changes |
| Plots | `analysis/results/plots/*.png` | Statistical test visualizations |

### 7.6 Expected Key Results (Statistics)

| Test | Statistic | p-value | Significant? |
|---|---|---|---|
| OLS Regression (group×condition) | F=9.82 | 1.7×10⁻¹⁰ | **Yes** |
| Chi-Square (text CF, 8 groups) | χ²=77.81 | 3.8×10⁻¹⁴ | **Yes** |
| Kruskal-Wallis (text CF, 8 groups) | H=77.64 | 4.2×10⁻¹⁴ | **Yes** |
| Wilcoxon (image GRL, paired) | W=0.0 | 0.063 | No (p>0.05) |

---

## 8. Pipeline 5 — Multi-Seed Robustness

### 8.1 What It Does

Retrains EfficientNet-B0 from scratch under 3 seeds [42, 123, 456] to estimate variance. This is the most time-consuming step (~5 hours with GPU).

### 8.2 Commands

```bash
# Full multi-seed (3 seeds × 3 conditions = 9 training runs)
python3 scripts/multi_seed_experiment.py

# Image-only (skip text/fusion)
python3 scripts/multi_seed_experiment.py --image-only

# Custom seeds
python3 scripts/multi_seed_experiment.py --seeds 42 123 456 789

# Smoke-test
python3 scripts/multi_seed_experiment.py --smoke-test
```

### 8.3 Output File

| File | Path | Description |
|---|---|---|
| Multi-seed results | `analysis/results/multi_seed_results.json` | Per-seed F1/AUC/FPR, means, stds, CIs |

### 8.4 Expected Key Results

| Condition | F1 Mean±Std | FPR Mean±Std |
|---|---|---|
| EfficientNet nCF | 0.7498 ± 0.0159 | 0.2538 ± 0.0401 |
| EfficientNet CF-no-adv | 0.7935 ± 0.0022 | 0.2785 ± 0.0202 |
| EfficientNet CF+GRL | 0.7845 ± 0.0028 | 0.2718 ± 0.0091 |

> **Note:** MiniLM multi-seed was never run. Use `mlp_cv_results.json` (bootstrap + CF size-ablation) for text model stability and capacity-confound evidence.

---

## 9. Pipeline 6 — Plot Generation & Comprehensive Evaluation

### 9.1 What It Does

Generates 13 publication-quality figures and produces the **master comprehensive evaluation JSON** (28 rows) that is the canonical source for all cross-model comparisons.

### 9.2 Prerequisites

Pipelines 1, 2, and 3 must be complete. The script reads:
- `text_models/enhanced_results/enhanced_results.json`
- `image_models/results/evaluation_results.json`
- `cross_modal/results/late_fusion_results.json`
- `image_models/results/training_log.txt`
- Cached MiniLM embeddings (`.npy` files)
- `data/datasets/final_dataset_18k.csv`

### 9.3 Command

```bash
python3 scripts/generate_all_plots.py
```

### 9.4 Output Files

| File | Path | Description |
|---|---|---|
| **Comprehensive evaluation** | `cross_modal/results/comprehensive_evaluation.json` | **THE canonical results table** — 28 rows, all models on unified 900-sample test set |
| Figures | `plots/figure_01_*.png` through `plots/figure_13_*.png` | Publication-quality 200 DPI figures |

### 9.5 Figures Generated

| Figure | Filename Pattern | Contents |
|---|---|---|
| 1 | `figure_01_*.png` | Training pipeline schematic |
| 2 | `figure_02_*.png` | Text model metric comparison |
| 3 | `figure_03_*.png` | Image model metric comparison |
| 4 | `figure_04_*.png` | ROC curves |
| 5 | `figure_05_*.png` | Fairness radar plots |
| 6 | `figure_06_*.png` | Fusion strategy comparison |
| 7 | `figure_07_*.png` | Per-group FPR bar charts |
| 8 | `figure_08_*.png` | t-SNE embedding visualizations |
| 9 | `figure_09_*.png` | Calibration curves |
| 10–13 | `figure_10_*` – `figure_13_*` | Additional analysis plots |

---

## 10. Pipeline 7 — Baseline Pipeline (Optional, CPU-Only)

This is an independent baseline using DenseNet-121 + TF-IDF (no transformers). It is not required to reproduce the main results but provides comparative context.

### 10.1 Environment

Requires a separate Python 3.12 environment with `baseline-pipeline/requirements.txt`.

### 10.2 Commands

```bash
cd baseline-pipeline

# Full run (3 seeds, both conditions)
python3 run.py

# Single seed
python3 run.py --seeds 42

# Text-only (skip image training)
python3 run.py --seeds 42 --skip-image

# MLP fusion only (requires existing text+image predictions)
python3 run_mlp_only.py
```

### 10.3 Configuration

All settings live in [`baseline-pipeline/config.py`](baseline-pipeline/config.py):

| Setting | Value | Notes |
|---|---|---|
| `SPLIT_RATIOS` | (0.60, 0.15, 0.25) | Different from main pipeline target (70/15/15; realized 69.99/15.00/15.01) |
| `SEEDS` | [42, 123, 456] | 3-seed runs |
| `TFIDF_MAX_FEATURES` | 10,000 | Word n-grams (1,2) |
| `IMAGE_SIZE` | 224 | Standard ImageNet input |
| `MAX_EPOCHS` | 7 | Max training epochs |
| `EARLY_STOP_PATIENCE` | 3 | |
| `MLP_FOLDS` | 5 | For Strategy B fusion |

### 10.4 Output

Results are written to `baseline-pipeline/results/`:
- `all_results.json` — aggregated results across seeds and conditions
- `*_test.csv` — per-sample prediction CSVs
- Statistical test outputs and result tables

---

## 11. Running via Docker

Docker provides a reproducible, self-contained environment:

```bash
# Build the container
docker build -t bias-eval .

# Run integration tests (default CMD)
docker run bias-eval

# Run a specific pipeline inside the container
docker run bias-eval python3 text_models/run_all.py --smoke-test

# Interactive shell
docker run -it bias-eval bash
```

The Dockerfile uses Python 3.10-slim, installs all dependencies, sets `PYTHONPATH=/app`, and caches HuggingFace models to `/app/.cache/`.

---

## 12. Smoke-Test Mode (Quick Validation)

Every major pipeline supports `--smoke-test` for fast validation (typically < 3 minutes):

```bash
python3 text_models/run_all.py --smoke-test
python3 image_models/run_all.py --smoke-test
python3 cross_modal/run_all.py --smoke-test
python3 analysis/run_all.py --smoke-test
python3 scripts/multi_seed_experiment.py --smoke-test
```

Smoke tests reduce sample counts (100 samples), training epochs (2 epochs), and bootstrap iterations (n=10). They verify pipeline machinery without waiting for convergence.

### Integration Tests

```bash
pytest tests/ -v --tb=short -x
```

Tests validate: dataset schema, prediction file formats, per-group ΔFPR structure, consistency results, model checksums, and cross-modal joinability.

---

## 13. Master Result File Reference

This table maps every number in `prof-report.md` to its authoritative JSON source and generator script.

### Text Models

| Result File | Generator Command | Key Metrics |
|---|---|---|
| `text_models/binary_fairness_results/binary_fairness_results.json` | `python3 text_models/binary_fairness_analysis.py` | TF-IDF LR/Ridge/NB/RF: F1, AUC, FPR per condition |
| `text_models/enhanced_results/enhanced_results.json` | `python3 text_models/enhanced_analysis.py` | TF-IDF + MiniLM (all classifiers) — **⚠️ MiniLM CF rows misaligned** |

### Image Models

| Result File | Generator Command | Key Metrics |
|---|---|---|
| `image_models/results/evaluation_results.json` | `python3 image_models/run_all.py` | EfficientNet nCF/CF-no-adv/CF+GRL: F1, AUC, FPR, per-group FPR, EO-diff |

### Cross-Modal Fusion

| Result File | Generator Command | Key Metrics |
|---|---|---|
| `cross_modal/results/comprehensive_evaluation.json` | `python3 scripts/generate_all_plots.py` | **Master table (28 rows)** — all models × conditions on unified test set |
| `cross_modal/results/late_fusion_results.json` | `python3 cross_modal/late_fusion_ensemble.py` | Late fusion F1, AUC, ECE, bootstrap 95% CI |
| `cross_modal/results/ablation_calibration_results.json` | `python3 cross_modal/ablation_calibration_study.py` | Weight sweep × calibration across 3 conditions; best ECE in summary table: 0.0174 (nCF) |
| `cross_modal/results/stacking_ensemble_results.json` | `python3 cross_modal/stacking_ensemble.py` | Stacking: train OOF diagnostics + held-out test metrics (no CV on test) |
| `cross_modal/results/learned_fusion_results.json` | `python3 cross_modal/learned_fusion.py` | 6 learned fusion strategies: F1, AUC || `cross_modal/results/cross_attention_fusion_results.json` | `python3 cross_modal/cross_attention_fusion.py` | GMU+GRL ensemble F1 by condition: nCF=0.860, CF-no-adv=0.830, CF+GRL=0.838 |

### Analysis & Statistics

| Result File | Generator Command | Key Metrics |
|---|---|---|
| `analysis/results/multi_seed_results.json` | `python3 scripts/multi_seed_experiment.py` | EfficientNet 3-seed: F1 mean±std, FPR mean±std |
| `analysis/results/mlp_cv_results.json` | `python3 analysis/mlp_cross_validation.py` | MiniLM+MLP bootstrap: F1=0.9463, CI [0.930, 0.960]; nCF CV; CF size-ablation curve |
| `analysis/results/enhanced_statistical_tests.json` | `python3 analysis/enhanced_statistical_tests.py` | OLS ANOVA, Chi-sq, Kruskal-Wallis, Cohen's d |
| `analysis/results/wilcoxon_results.json` | `python3 analysis/statistical_tests.py` | Wilcoxon: image GRL W=0.0, p=0.0625 |

---

## 14. JSON Output Schemas

This section documents the structure of each canonical result file so that reviewers, AI agents, or downstream scripts can parse them programmatically.

### 14.1 `text_models/binary_fairness_results/binary_fairness_results.json`

```json
{
  "overall_metrics": [
    {
      "name": "logistic_regression",    // Model name
      "condition": "ncf",               // "ncf" | "cf"
      "training_time": 0.45,            // seconds
      "accuracy": 0.812,
      "precision": 0.795,
      "recall": 0.834,
      "f1": 0.814,
      "roc_auc": 0.882,
      "avg_prec": 0.871,
      "brier": 0.142,
      "fpr": 0.198,                     // False Positive Rate
      "fnr": 0.166,                     // False Negative Rate
      "tpr": 0.834,
      "tnr": 0.802,
      "tp": 375, "fp": 89, "fn": 74, "tn": 354,
      "ci": {
        "accuracy": [0.790, 0.834],     // Bootstrap 95% CI
        "f1": [0.790, 0.838],
        "roc_auc": [0.860, 0.903],
        "fpr": [0.162, 0.236],
        "fnr": [0.131, 0.203]
      }
    }
    // ... 8 total entries (4 models × 2 conditions)
  ]
}
```

**Key extraction:** `results["overall_metrics"]` → filter by `name` and `condition`.

### 14.2 `text_models/enhanced_results/enhanced_results.json`

```json
{
  "results": [
    {
      "name": "lr_tfidf",              // Model identifier
      "condition": "ncf",               // "ncf" | "cf"
      "training_time": 0.32,
      "default_accuracy": 0.812,        // Default threshold (0.5) metrics
      "default_precision": 0.795,
      "default_recall": 0.834,
      "default_f1": 0.814,
      "default_fpr": 0.198,
      "default_fnr": 0.166,
      "default_tp": 375, "default_fp": 89, "default_fn": 74, "default_tn": 354,
      "opt_accuracy": 0.818,            // Optimized threshold metrics
      "opt_precision": 0.801,
      "opt_recall": 0.838,
      "opt_f1": 0.819,
      "opt_fpr": 0.192,
      "opt_fnr": 0.162,
      "opt_threshold": 0.42,            // Val-tuned threshold
      "roc_auc": 0.882,
      "avg_prec": 0.871,
      "brier": 0.142,
      "ci": {
        "f1": [0.790, 0.838],
        "roc_auc": [0.860, 0.903],
        "fpr": [0.162, 0.236],
        "opt_f1": [0.795, 0.843]
      }
    }
    // ... 16 total entries (8 models × 2 conditions)
  ]
}
```

**Model names:** `lr_tfidf`, `svm_tfidf`, `rf_tfidf`, `ridge_tfidf`, `lr_minilm`, `svm_minilm`, `mlp_minilm`, `nb_tfidf`

### 14.3 `image_models/results/evaluation_results.json`

```json
{
  "ncf": {
    "threshold": 0.24,                  // Optimal threshold
    "metrics": {
      "accuracy": 0.748,
      "precision": 0.693,
      "recall": 0.895,
      "f1": 0.781,
      "auc_roc": 0.832,
      "brier": 0.173,
      "fpr": 0.401,
      "fnr": 0.105
    },
    "fairness": {
      "demographic_parity_diff": 0.448,
      "equalized_odds_diff": 0.680,
      "eo_fpr_component": 0.576,
      "eo_tpr_component": 0.104
    },
    "per_group": {
      "race/ethnicity": {
        "fpr": 0.333, "fnr": 0.200,
        "n": 150, "n_non_hate": 75, "n_hate": 75
      },
      "religion": { "fpr": 0.267, "fnr": 0.187, "n": 150, "...": "..." },
      "gender": { "...": "..." },
      "sexual_orientation": { "...": "..." },
      "national_origin/citizenship": { "...": "..." },
      "disability": { "...": "..." },
      "age": { "...": "..." },
      "multiple/none": { "...": "..." }
    },
    "bootstrap_ci": {
      "accuracy": { "mean": 0.785, "ci_lower": 0.761, "ci_upper": 0.809 },
      "f1": { "mean": 0.771, "ci_lower": 0.744, "ci_upper": 0.798 },
      "auc_roc": { "mean": 0.817, "ci_lower": 0.790, "ci_upper": 0.844 }
    }
  },
  "cf_no_adv": { "...": "same structure as ncf" },
  "cf": { "...": "same structure as ncf (this is CF+GRL)" },
  "mcnemar_tests": {
    "ncf_vs_cf_no_adv": { "statistic": 12.5, "p_value": 0.0004, "significant": true },
    "ncf_vs_cf": { "...": "..." },
    "cf_no_adv_vs_cf": { "...": "..." }
  }
}
```

**Key extraction:** `results["ncf"]["metrics"]["f1"]` for nCF F1; `results["cf"]["per_group"]["disability"]["fpr"]` for per-group FPR.

### 14.4 `cross_modal/results/comprehensive_evaluation.json`

```json
{
  "description": "Comprehensive evaluation: all models × all conditions",
  "n_rows": 28,
  "results": [
    {
      "modality": "Text",               // "Text" | "Image" | "Fusion"
      "model": "MiniLM + MLP",          // Model display name
      "condition": "CF",                 // "NCF" | "CF"
      "accuracy": 0.958,
      "precision": 0.954,
      "recall": 0.962,
      "macro_f1": 0.956,                // THE canonical F1 value
      "auc_roc": 0.979,
      "fpr": 0.059,
      "fnr": 0.038,
      "brier": 0.035,
      "threshold": 0.45,
      "bias_delta_fpr": -0.178,          // Change vs nCF baseline (negative = improved)
      "bias_direction": "DECREASED"      // "BASELINE" | "INCREASED" | "DECREASED"
    }
    // ... 28 total entries
  ]
}
```

**This is the canonical source for the text models table in the report.** Use this instead of `enhanced_results.json` for MiniLM CF numbers.

**Key extraction:** Filter `results` array by `modality` and `condition`:
```python
import json
with open("cross_modal/results/comprehensive_evaluation.json") as f:
    data = json.load(f)
text_cf = [r for r in data["results"] if r["modality"] == "Text" and r["condition"] == "CF"]
```

### 14.5 `cross_modal/results/late_fusion_results.json`

```json
{
  "description": "Cross-Modal Late Fusion",
  "text_model": "minilm_mlp_cf.joblib",
  "image_model": "efficientnet_cf_no_adv.pth",
  "bootstrap_n": 1500,
  "table6": [
    {
      "model": "Text-Only",              // Row label
      "f1": 0.956, "auc_roc": 0.979,
      "fpr": 0.059, "fnr": 0.038,
      "precision": 0.954, "recall": 0.962,
      "eo_diff": 0.312,                  // Equalized Odds diff
      "ece": 0.058,                      // Expected Calibration Error
      "f1_95ci": [0.940, 0.970]          // Bootstrap 95% confidence interval
    },
    { "model": "Image-Only", "...": "..." },
    { "model": "Equal Fusion", "...": "..." },
    { "model": "Learned Fusion", "...": "..." },
    { "model": "  ],
  "detailed_results": {
    "text_only": {
      "metrics": { "accuracy": 0.958, "f1": 0.956, "...": "..." },
      "fairness": { "dp_diff": 0.250, "eo_diff": 0.312 },
      "per_group": {
        "race/ethnicity": { "fpr": 0.040, "fnr": 0.027, "n": 150 },
        "...": "... (8 groups)"
      }
    },
    "image_only": { "...": "..." },
    "equal_fusion": { "...": "..." }
  }
}
```

### 14.6 `cross_modal/results/stacking_ensemble_results.json`

```json
{
  "description": "Stacking Ensemble: train OOF fitting + single held-out test evaluation",
  "best_meta_learner": "LogisticRegression",
  "n_train_samples": 4158,
  "n_test_samples": 892,
  "n_folds": 5,
  "n_meta_features": 9,
  "feature_names": ["p_text", "p_image", "p_text*p_image", "|p_text-p_image|",
                     "max(pt,pi)", "min(pt,pi)", "mean(pt,pi)", "p_text^2", "p_image^2"],
  "meta_learner_selection": {
    "LogisticRegression": { "mean_f1": 0.936, "std_f1": 0.029 },
    "GradientBoosting": { "mean_f1": 0.930, "std_f1": 0.032 },
    "MLP(32,16)": { "mean_f1": 0.928, "std_f1": 0.035 },
    "ExtraTrees": { "mean_f1": 0.925, "std_f1": 0.038 }
  },
  "fold_f1s": [0.946, 0.944, 0.931, 0.961, 0.879],
  "threshold_train_oof": 0.615,
  "oof_metrics_uncalibrated": {
    "accuracy": 0.938, "precision": 0.935, "recall": 0.941,
    "f1": 0.936, "auc_roc": 0.973, "brier": 0.052,
    "fpr": 0.082, "fnr": 0.059, "ece": 0.034
  },
  "test_metrics_uncalibrated": {
    "accuracy": 0.936, "precision": 0.932, "recall": 0.940,
    "f1": 0.936, "auc_roc": 0.972, "brier": 0.055,
    "fpr": 0.069, "fnr": 0.060, "ece": 0.014
  },
  "test_metrics_calibrated": { "...": "same structure, isotonic applied" },
  "per_group_fpr_stacking_test": {
    "race/ethnicity": { "n": 150, "n_non_hate": 75, "fpr": 0.053 },
    "...": "... (8 groups)"
  }
}
```

### 14.7 `cross_modal/results/cross_attention_fusion_results.json`

```json
{
  "description": "Feature-Level Cross-Modal Fusion: GMU + Cross-Attention (...)",
  "architecture": {
    "text_dim": 384,
    "image_raw_dim": 1280,
    "image_proj_dim": 384,
    "cross_attention_output_components": [
      "text_attn", "image_attn", "gated_text", "gated_image"
    ],
    "concat_dim": 1536,
    "concat_formula": "concat_dim = 4 * text_dim = 1536",
    "task_head": "Linear(1536,256) -> ReLU -> Dropout -> Linear(256,1)",
    "adv_head": "GRL -> Linear(1536,256) -> ReLU -> Linear(256,8)"
  },
  "parameter_sanity": {
    "derived_concat_dim": 1536,
    "model_concat_dim": 1536,
    "concat_dim_matches": true,
    "trainable_params": 2761481,
    "total_params": 2761481
  },
  "n_train": 4158,
  "n_val": 891,
  "n_test": 892,
  "n_folds": 5,
  "device": "cpu",
  "smoke_test": false,
  "cv_results": {
    "fold_val_f1s": [0.8777, 0.8769, 0.8774, 0.8836, 0.8654],
    "mean_f1": 0.8762,
    "std_f1": 0.0059,
    "mean_auc": 0.9199,
    "std_auc": 0.0071
  },
  "ensemble": {
    "metrics": { "f1": 0.8550, "auc_roc": 0.9204, "fpr": 0.2185 },
    "fairness": { "demographic_parity_diff": 0.7810, "equalised_odds_diff": 0.5978 },
    "ece": 0.0383
  },
  "final_model": {
    "metrics": { "f1": 0.8019, "auc_roc": 0.8908, "fpr": 0.1486 },
    "ece": 0.1026
  },
  "runtime_seconds": 508.7
}
```

### 14.8 `analysis/results/multi_seed_results.json`

```json
{
  "metadata": {
    "seeds": [42, 123, 456],
    "n_seeds": 3,
    "timestamp": "2026-03-10T14:32:00",
    "total_time_seconds": 16920.5,
    "total_time_human": "4.7h"
  },
  "models": {
    "EfficientNet (ncf)": {
      "n_runs": 3,
      "per_seed": [
        { "f1": 0.770, "auc": 0.816, "fpr": 0.387, "fnr": 0.199, "seed": 42, "time": 1820.3 },
        { "f1": 0.738, "auc": 0.801, "fpr": 0.210, "fnr": 0.264, "seed": 123, "time": 1795.1 },
        { "f1": 0.741, "auc": 0.808, "fpr": 0.164, "fnr": 0.251, "seed": 456, "time": 1803.7 }
      ],
      "f1": { "mean": 0.7498, "std": 0.0159, "min": 0.738, "max": 0.770, "ci_95": [0.710, 0.790], "values": [0.770, 0.738, 0.741] },
      "auc": { "mean": 0.808, "std": 0.008, "...": "..." },
      "fpr": { "mean": 0.2538, "std": 0.0401, "...": "..." },
      "fnr": { "mean": 0.238, "std": 0.034, "...": "..." }
    },
    "EfficientNet (cf_no_adv)": { "...": "same structure" },
    "EfficientNet (cf)": { "...": "same structure (CF+GRL)" }
  }
}
```

### 14.9 `analysis/results/mlp_cv_results.json`

```json
{
  "experiment": "MiniLM+MLP Validation & CF Augmentation Analysis",
  "model": "sentence-transformers/all-MiniLM-L12-v2 -> MLP(256,128)",
  "part1_bootstrap": {
    "saved_model_f1": 0.94633,
    "n_bootstrap": 1500,
    "f1_95ci": [0.92997, 0.96019],
    "f1_std": 0.0078
  },
  "part2_cv_ncf": {
    "n_folds": 5,
    "f1_mean": 0.68034,
    "f1_std": 0.018,
    "fold_details": [
      { "fold": 0, "f1": 0.692, "auc_roc": 0.745, "accuracy": 0.701, "fpr": 0.310, "time_s": 4.2 },
      "..."
    ]
  },
  "part3_retrain": {
    "ncf_canonical_f1": 0.8633,
    "cf_saved_f1": 0.94589,
    "cf_augmentation_delta_f1": 0.08259,
    "cf_augmentation_pct": 9.6,
    "note": "nCF sourced from cross_modal/results/comprehensive_evaluation.json; CF uses saved-model bootstrap mean"
  },
  "part4_learning_curve": {
    "0.1":  { "f1_mean": 0.520, "f1_std": 0.045, "auc_mean": 0.580, "auc_std": 0.040, "n_train_mean": 415 },
    "0.25": { "f1_mean": 0.580, "f1_std": 0.032, "...": "..." },
    "0.5":  { "...": "..." },
    "0.75": { "...": "..." },
    "1.0":  { "f1_mean": 0.680, "f1_std": 0.018, "...": "..." }
  },
  "part5_cf_size_ablation": {
    "description": "Capacity confound check via CF subset scaling",
    "targets_full_dataset_sizes": [6000, 9000, 12000, 18000],
    "train_rows_total_cf_split": 12470,
    "n_repeats_per_size": 3,
    "results": {
      "6000": {
        "full_data_target": 6000,
        "train_rows": 4157,
        "f1_mean": 0.90,
        "f1_std": 0.01
      },
      "18000": {
        "full_data_target": 18000,
        "train_rows": 12470,
        "f1_mean": 0.94,
        "f1_std": 0.01
      }
    }
  }
}
```

### 14.10 `analysis/results/enhanced_statistical_tests.json`

```json
{
  "chi_squared_test": {
    "text_cf": { "chi2": 77.81, "p_value": 3.84e-14, "df": 7, "significant": true },
    "fusion": { "chi2": 35.70, "p_value": 8.26e-6, "df": 7, "significant": true }
  },
  "kruskal_wallis_test": {
    "text_cf": { "H_statistic": 77.64, "p_value": 4.17e-14, "significant": true }
  },
  "logistic_regression_anova": {
    "F_statistic": 9.817,
    "p_value": 1.72e-10,
    "significant": true
  },
  "per_group_fpr": {
    "race/ethnicity": {
      "fpr_nCF": 0.240, "fpr_CF": 0.053,
      "ci_lower": 0.018, "ci_upper": 0.120,
      "n_samples": 150, "n_non_hate": 75
    },
    "religion": { "...": "..." },
    "...": "... (8 groups total)"
  },
  "effect_size_cohens_d": {
    "overall": 0.85,
    "by_group": { "race/ethnicity": 1.2, "religion": 0.9, "...": "..." }
  },
  "pairwise_tests": {
    "race_vs_religion": { "U_statistic": 142, "p_value_raw": 0.023, "p_value_corrected": 0.069 },
    "...": "... (all pairs)"
  }
}
```

### 14.11 `analysis/results/wilcoxon_results.json`

```json
{
  "image_grl_paired": {
    "test": "Wilcoxon signed-rank",
    "comparison": "EfficientNet nCF vs CF+GRL per-group FPR",
    "W_statistic": 0.0,
    "p_value": 0.0625,
    "n_groups": 8,
    "significant": false,
    "note": "p > 0.05 — NOT significant at α=0.05"
  }
}
```

### 14.12 `cross_modal/results/ablation_calibration_results.json`

```json
{
  "description": "Weight ablation × calibration study",
  "weight_sweep": {
    "fusion_w0.00": { "f1": 0.890, "auc": 0.937, "ece": 0.044, "dp_diff": 0.744 },
    "fusion_w0.05": { "...": "..." },
    "...": "... (w from 0.0 to 1.0 in steps of 0.05)",
    "fusion_w1.00": { "f1": 0.875, "auc": 0.946, "ece": 0.033, "dp_diff": 0.755 }
  },
  "calibration": {
    "fusion_w0.50_none": { "f1": 0.907, "ece": 0.066, "dp_diff": 0.739 },
    "fusion_w0.50_temperature_scaling": { "f1": 0.912, "ece": 0.017, "dp_diff": 0.700 },
    "fusion_w0.50_isotonic_regression": { "f1": 0.915, "ece": 0.019, "dp_diff": 0.739 }
  },
  "best_config": "fusion_w0.50_isotonic_regression",
  "best_ece": 0.0174
}
```

---

## 15. Result Extraction Guide

### 15.1 Python Quick-Access Snippets

**Extract text model results from the canonical source:**

```python
import json

# Load the master table
with open("cross_modal/results/comprehensive_evaluation.json") as f:
    data = json.load(f)

# Get MiniLM + MLP CF results (THE canonical text result)
mlp_cf = next(r for r in data["results"]
              if r["model"] == "MiniLM + MLP" and r["condition"] == "CF")
print(f"F1={mlp_cf['macro_f1']}, AUC={mlp_cf['auc_roc']}, FPR={mlp_cf['fpr']}")
# Expected: F1=0.956, AUC=0.979, FPR=0.059
```

**Extract image model per-group FPR:**

```python
with open("image_models/results/evaluation_results.json") as f:
    img = json.load(f)

for condition in ["ncf", "cf_no_adv", "cf"]:
    print(f"\n--- {condition} ---")
    print(f"F1: {img[condition]['metrics']['f1']}")
    print(f"Overall FPR: {img[condition]['metrics']['fpr']}")
    for group, stats in img[condition]["per_group"].items():
        print(f"  {group}: FPR={stats['fpr']:.3f} (n={stats['n']})")
```

**Extract fusion strategy comparison:**

```python
with open("cross_modal/results/late_fusion_results.json") as f:
    fusion = json.load(f)

for row in fusion["table6"]:
    print(f"{row['model']:25s}  F1={row['f1']:.3f}  AUC={row['auc_roc']:.3f}  "
          f"ECE={row['ece']:.4f}
```

**Extract multi-seed variance:**

```python
with open("analysis/results/multi_seed_results.json") as f:
    seeds = json.load(f)

for model_name, model_data in seeds["models"].items():
    f1 = model_data["f1"]
    fpr = model_data["fpr"]
    print(f"{model_name}: F1={f1['mean']:.4f}±{f1['std']:.4f}, "
          f"FPR={fpr['mean']:.4f}±{fpr['std']:.4f}")
```

**Extract statistical significance:**

```python
with open("analysis/results/enhanced_statistical_tests.json") as f:
    stats = json.load(f)

chi2 = stats["chi_squared_test"]["text_cf"]
print(f"Chi-square: χ²={chi2['chi2']:.2f}, p={chi2['p_value']:.2e}")

anova = stats["logistic_regression_anova"]
print(f"OLS ANOVA: F={anova['F_statistic']:.3f}, p={anova['p_value']:.2e}")
```

**Extract bootstrap CI for MiniLM+MLP:**

```python
with open("analysis/results/mlp_cv_results.json") as f:
    cv = json.load(f)

boot = cv["part1_bootstrap"]
print(f"F1={boot['saved_model_f1']:.4f}, 95% CI={boot['f1_95ci']}")
# Expected: F1=0.9463, 95% CI=[0.930, 0.960]
```

### 15.2 jq Command-Line Extraction

For users who prefer shell-based extraction:

```bash
# MiniLM + MLP CF F1 from comprehensive evaluation
jq '.results[] | select(.model == "MiniLM + MLP" and .condition == "CF") | .macro_f1' \
  cross_modal/results/comprehensive_evaluation.json

# All image model F1 values
jq '{ncf: .ncf.metrics.f1, cf_no_adv: .cf_no_adv.metrics.f1, cf_grl: .cf.metrics.f1}' \
  image_models/results/evaluation_results.json

# Late fusion F1 values
jq '.table6[] | {model: .model, f1: .f1, ece: .ece}' \
  cross_modal/results/late_fusion_results.json

# Multi-seed means
jq '.models | to_entries[] | {model: .key, f1_mean: .value.f1.mean, f1_std: .value.f1.std}' \
  analysis/results/multi_seed_results.json

# Statistical test p-values
jq '{anova_p: .logistic_regression_anova.p_value, chi2_p: .chi_squared_test.text_cf.p_value}' \
  analysis/results/enhanced_statistical_tests.json
```

---

## 16. Troubleshooting

### 16.1 Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'canonical_splits'` | `PYTHONPATH` not set | `export PYTHONPATH="$(pwd):$PYTHONPATH"` |
| `FileNotFoundError: data/datasets/final_dataset_18k.csv` | Dataset not present | Ensure the CSV is at the expected path |
| `FileNotFoundError` for image directories | Generated images missing | Verify `Hate/` and `non-hate/` directories exist with PNGs |
| MiniLM model download fails | No internet / HuggingFace hub down | Pre-download: `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L12-v2')"` |
| `RuntimeError: CUDA out of memory` | GPU VRAM exceeded | Reduce `--batch-size` or use `--device cpu` |
| Slow image pipeline on CPU | Expected (~5h per condition) | Use `--device cuda` if GPU available |
| `Missing train fusion prediction file: fusion_train_oof_predictions.csv` | Leakage-safe stacking requires explicit train OOF input | Generate `cross_modal/results/predictions/fusion_train_oof_predictions.csv` before running `cross_modal/stacking_ensemble.py` |
| `pyenv: command not found` | pyenv not installed (baseline only) | Install pyenv or use Docker for baseline pipeline |
| Different numbers from report | Wrong JSON file consulted | Always use `comprehensive_evaluation.json` for text; see §17 caveats |
| `KeyError` when parsing JSON | Schema changed between runs | Re-run the generator script to produce fresh output |

### 16.2 Verifying Output Integrity

After running all pipelines, verify that all expected output files exist:

```bash
# Check all canonical result files exist
for f in \
  text_models/binary_fairness_results/binary_fairness_results.json \
  text_models/enhanced_results/enhanced_results.json \
  image_models/results/evaluation_results.json \
  cross_modal/results/comprehensive_evaluation.json \
  cross_modal/results/late_fusion_results.json \
  cross_modal/results/ablation_calibration_results.json \
  cross_modal/results/stacking_ensemble_results.json \
  cross_modal/results/learned_fusion_results.json \
  cross_modal/results/  cross_modal/results/cross_attention_fusion_results.json \
  analysis/results/multi_seed_results.json \
  analysis/results/mlp_cv_results.json \
  analysis/results/enhanced_statistical_tests.json \
  analysis/results/wilcoxon_results.json; do
  if [ -f "$f" ]; then
    echo "OK   $f"
  else
    echo "MISSING  $f"
  fi
done
```

### 16.3 Running Integration Tests

```bash
pytest tests/ -v --tb=short -x
```

Tests check: dataset schema (18K rows, required columns), prediction file formats (~892 rows), per-group ΔFPR structure, consistency results, model checksum integrity, and cross-modal joinability (text and image CSVs share all `counterfactual_id`s).

---

## 17. Known Caveats & Limitations

### 17.1 Test Set Alignment

`text_models/enhanced_results/enhanced_results.json` contains MiniLM CF results evaluated on a **misaligned test set**, producing artificially low F1 (~0.56–0.63). **Always use `cross_modal/results/comprehensive_evaluation.json`** for canonical MiniLM CF metrics (F1=0.956 on the unified 900-sample test set).

#
### 17.3 Multi-Seed Coverage

Only EfficientNet has multi-seed results (3 seeds: [42, 123, 456]). MiniLM multi-seed was **never run**. For MiniLM robustness, use `mlp_cv_results.json`: bootstrap CI (1,500 resamples) plus CF size-ablation (6K/9K/12K/18K equivalents) to address the data-volume/step-count confound.

### 17.4 Single-Seed vs Multi-Seed FPR

The image model FPR in Section 3 of `prof-report.md` (e.g., nCF FPR=0.401) is from seed 42 only. The multi-seed mean (Section 7) is nCF FPR=0.2538±0.0401 — substantially lower because seed 42 is an above-average FPR run.

### 17.5 Fusion Image Component

Fusion outputs are now condition-aware and generated for all three image settings (`nCF`, `CF-no-adv`, `CF+GRL`). For this rerun, CF-no-adv remains the strongest CF variant for several fusion strategies, while nCF and CF+GRL are explicitly included in consolidated comparison artifacts.

### 17.6 Deprecated Script

**`training/train_text_models.py` is deprecated** — it contains a data-leakage bug. All text model work should use scripts in the `text_models/` directory.

### 17.7 Baseline Pipeline Split Ratio

The baseline pipeline (`baseline-pipeline/`) uses a (60/15/25) train/val/test split ratio, which differs from the main pipeline's target split (70/15/15; realized 69.99/15.00/15.01). Results are not directly comparable without accounting for this difference.

### 17.8 GRL Adversarial Weight Sensitivity

The GRL adversarial weight was set to `0.5` for the image CF+GRL pipeline. Sensitivity to this parameter was not systematically evaluated. Values in range `[0.3, 0.7]` were informally tested during development.

For cross-attention fusion (`cross_modal/cross_attention_fusion.py`), the GRL adversarial-loss weight is `0.3`.

---

## Quick-Reference: Full Reproduction in One Block

```bash
# 0. Setup
cd /path/to/major-project
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 1. Validate splits
python3 scripts/validate_canonical_splits.py

# 2. Text models
python3 text_models/run_all.py

# 3. Image models (use --device cuda if GPU available)
python3 image_models/run_all.py --device cpu

# 4. Cross-modal fusion (late fusion + ablation run separately)
python3 cross_modal/late_fusion_ensemble.py
python3 cross_modal/ablation_calibration_study.py
python3 cross_modal/run_all.py

# 5. Statistical analysis
python3 analysis/mlp_cross_validation.py
python3 analysis/enhanced_statistical_tests.py
python3 analysis/statistical_tests.py

# 6. Multi-seed robustness (long-running, ~5h with GPU)
python3 scripts/multi_seed_experiment.py

# 7. Generate plots + comprehensive evaluation JSON
python3 scripts/generate_all_plots.py

# 8. Verify
pytest tests/ -v --tb=short -x
```

---

*This document was generated from the project source code and [`prof-report.md`](prof-report.md). For questions about the experimental design, see [`CLAUDE.md`](CLAUDE.md) and [`docs/REPORT.md`](docs/REPORT.md).*
