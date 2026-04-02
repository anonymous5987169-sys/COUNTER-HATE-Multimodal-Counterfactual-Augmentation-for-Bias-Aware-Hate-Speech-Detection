# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ML research project studying whether Counterfactual Data Augmentation (CAD) introduces or amplifies bias in hate speech detection systems. Target venue: ACM Multimedia 2026. The pipeline evaluates fairness across 8 protected identity groups under two experimental conditions (nCF = no counterfactuals, CF = with counterfactuals).

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run integration tests
```bash
pytest tests/ -v --tb=short -x
# or via Docker:
docker build -t bias-eval . && docker run bias-eval
```

### Text models pipeline (canonical)
```bash
python3 text_models/run_all.py
python3 text_models/run_all.py --smoke-test   # fast validation
```

### Image/training pipeline
```bash
python training/run_pipeline.py
python training/train_image_models.py --model all --condition all
python training/evaluate_bias.py --model all
```

### Baseline pipeline (CPU-only, Python 3.12 venv)
```bash
python baseline-pipeline/run.py                         # Full run, 3 seeds
python baseline-pipeline/run.py --seeds 42              # Single seed
python baseline-pipeline/run.py --seeds 42 --skip-image # Text-only
python baseline-pipeline/run_mlp_only.py
```

### Cross-modal fusion
```bash
python3 cross_modal/run_all.py
python3 cross_modal/run_all.py --smoke-test
python3 cross_modal/cross_attention_fusion.py --smoke-test
```

### Statistical analysis
```bash
python3 analysis/run_all.py
python3 analysis/run_all.py --smoke-test
```

## Architecture

### Experimental Conditions
- **nCF**: 6,000 original text samples only
- **CF**: 18,000 samples (6K originals + 12K LLM/regex-generated counterfactuals via identity-term substitution)

### Dataset
- `data/datasets/final_dataset_18k.csv` — main dataset (18K rows)
- `canonical_splits.py` — **single source of truth** for train/val/test splits (target 70/15/15; realized 69.99/15.00/15.01 with counts 4,158/891/892, stratified on `class_label`, split at `original_sample_id` level, `random_state=42`). Val/test sets contain originals only.
- 8 classes (750 original samples each): `hate_race`, `hate_religion`, `hate_gender`, `hate_other`, `offensive_non_hate`, `neutral_discussion`, `counter_speech`, `ambiguous`

### Pipeline Modules

| Module | Location | Purpose |
|---|---|---|
| Dataset construction | `src/counterfactual_gen/` | Builds dataset from Kennedy et al. 2020 (`ucberkeley-dlab/measuring-hate-speech`) |
| Text models (canonical) | `text_models/` | TF-IDF+LR/SVM/RF and MiniLM+LR/SVM/MLP |
| Image models | `training/` | EfficientNet-B0, ResNet-50, ViT — with optional GRL adversarial head |
| Baseline (CPU-only) | `baseline-pipeline/` | DenseNet-121 + TF-IDF, no transformers, max 7 epochs |
| Cross-modal fusion | `cross_modal/` | Late fusion, stacking, learned, cross-attention GMU |
| Analysis | `analysis/` | Statistical tests, calibration, intersectional bias, per-group fairness |
| Generated images | `Hate/*/generated_images/`, `non-hate/*/` | ~18K PNGs from Z-Image-Turbo T2I |

### Text → Image → Fusion Data Flow
1. Text: TF-IDF (10K features, ngram 1–2) or MiniLM-L12-v2 sentence embeddings (384-dim)
2. Image: EfficientNet-B0 backbone (1280-dim → projected to 384-dim)
3. Fusion strategies: late fusion (score avg, w=0.50), stacking ensemble (meta LR/SVM/MLP, 5-fold CV), learned fusion (6 strategies), cross-attention GMU with bidirectional attention + GRL
4. Calibration: isotonic regression or temperature scaling
5. Evaluation: F1, AUC, FPR, FNR, ECE, EO-diff; per-group FPR across 8 identity groups

### Cross-Attention Fusion Model (`cross_modal/cross_attention_fusion.py`)
- Text: MiniLM-L12-v2 (frozen) → 384-dim
- Image: EfficientNet-B0 (frozen) → 1280 → Linear → 384-dim
- GMU gate + bidirectional cross-attention → concat [text_attn; image_attn; text_orig; image_proj] (1536-dim) → classifier
- Adversarial head (GRL) for fairness debiasing
- Cross-attention GRL adversarial-loss weight: 0.3
- Image CF+GRL pipeline adversarial-loss weight: 0.5 (`image_models/config.py`, `ADV_WEIGHT`)
- Image GRL sensitivity was not systematically tuned; [0.3, 0.7] was informally tested.
- AdamW + CosineAnnealingLR, early stopping (patience=7), label smoothing=0.05, 5-fold CV

## Canonical Result Files

Every number in `prof-report.md` traces back to one of these files. The table below lists the authoritative path, what it contains, and the exact script that generates it.

### Text Models
| File | Contents | Generator |
|---|---|---|
| `text_models/binary_fairness_results/binary_fairness_results.json` | Binary TF-IDF results: LR, Ridge, NB, RF — F1/AUC/FPR per condition | `python3 text_models/binary_fairness_analysis.py` |
| `text_models/enhanced_results/enhanced_results.json` | Binary TF-IDF + MiniLM all classifiers. **Note:** MiniLM CF results here are evaluated on an unaligned test set and are incorrect — use `cross_modal/results/comprehensive_evaluation.json` instead | `python3 text_models/enhanced_analysis.py` |

### Image Models
| File | Contents | Generator |
|---|---|---|
| `image_models/results/evaluation_results.json` | **Canonical image results** — EfficientNet nCF / CF-no-adv / CF+GRL: F1, AUC, FPR, per-group FPR across 8 identity groups, EO-diff | `python3 image_models/run_all.py` (calls `image_models/evaluate.py`) |

### Cross-Modal Fusion
| File | Contents | Generator |
|---|---|---|
| `cross_modal/results/comprehensive_evaluation.json` | **Master text + image + fusion table** (28 rows): all models × all conditions on the unified 900-sample fusion test set. **Canonical source for the text models table in the report.** | `python3 scripts/generate_all_plots.py` |
| `cross_modal/results/late_fusion_results.json` | Late fusion table6: Text-Only, Image-Only, Equal/Learned fusion — F1, AUC, ECE, bootstrap 95% CI | `python3 cross_modal/late_fusion_ensemble.py` |
| `cross_modal/results/ablation_calibration_results.json` | Weight ablation w=[0.0…1.0] × calibration method (none / temp-scaling / isotonic): F1, AUC, ECE. Best config: `fusion_w0.50_isotonic_regression` (ECE=0.0143) | `python3 cross_modal/ablation_calibration_study.py` |
| `cross_modal/results/stacking_ensemble_results.json` | Stacking meta-learner: train-only 5-fold OOF for fitting/tuning + single held-out test evaluation (no CV on test) | `python3 cross_modal/stacking_ensemble.py` |
| `cross_modal/results/learned_fusion_results.json` | 6 learned fusion strategies (equal_weight, product, etc.): F1, AUC per strategy | `python3 cross_modal/learned_fusion.py` |
| `cross_modal/results/cross_attention_fusion_results.json` | Cross-attention GMU + GRL: 5-fold CV mean F1=0.876±0.006, AUC=0.920±0.007, ensemble F1=0.855, EO-diff=0.781 | `python3 cross_modal/cross_attention_fusion.py` |

### Analysis & Statistics
| File | Contents | Generator |
|---|---|---|
| `analysis/results/multi_seed_results.json` | EfficientNet **only** — 3 seeds [42, 123, 456]: nCF F1=0.7498±0.0159, CF-no-adv F1=0.7935±0.0022, CF+GRL F1=0.7845±0.0028. MiniLM multi-seed was **never run**. | `python3 scripts/multi_seed_experiment.py` |
| `analysis/results/mlp_cv_results.json` | MiniLM+MLP CF bootstrap (n=1500) on fusion_test_900: F1=0.9463, 95% CI [0.930, 0.960], std=0.0078. Also nCF 5-fold CV (F1=0.680±0.018), learning curve, and CF size-ablation (6K/9K/12K/18K) for capacity-confound checks. | `python3 analysis/mlp_cross_validation.py` |
| `analysis/results/enhanced_statistical_tests.json` | OLS ANOVA (F=9.817, p=1.72×10⁻¹⁰), Chi-square (text_cf χ²=77.81 p=3.84×10⁻¹⁴; fusion χ²=35.70 p=8.26×10⁻⁶), Kruskal-Wallis (text_cf H=77.64 p=4.17×10⁻¹⁴), per-group FPR, Cohen's d | `python3 analysis/enhanced_statistical_tests.py` |
| `analysis/results/wilcoxon_results.json` | Wilcoxon paired test: image GRL W=0.0, p=0.0625 (NOT significant); text MiniLM unpaired not available (prediction CSVs missing) | `python3 analysis/statistical_tests.py` |

### Important Caveats
- **Test set alignment**: All fusion strategies (text, image, late) use the correctly aligned `fusion_test_900`.
- **MiniLM robustness**: No multi-seed run exists. Use `mlp_cv_results.json` (bootstrap + CF size-ablation) for MiniLM stability and capacity-confound evidence.

## Important Notes

- **`training/train_text_models.py` is deprecated** — has a data-leakage bug. Use `text_models/` for all text model work.
- Multiple Python virtual environments coexist: `.venv/` (Python 3.14, main), `src/counterfactual_gen/venv/` (Python 3.13), and baseline-pipeline uses a pyenv 3.12 venv.
- Image paths are hardcoded in `baseline-pipeline/config.py` pointing to `Hate/` and `non-hate/` directories.
- Results, models, and predictions are written to `*/results/`, `*/models/`, and `baseline-pipeline/results/` — these are large and should not be committed.
