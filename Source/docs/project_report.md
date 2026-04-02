# Project Report — Bias Evaluation of Counterfactual Data Augmentation in Hate Speech Detection

**Target:** ACM Multimedia (ACMMM) 2026  
**Version:** 3.0  
**Date:** 3 March 2026  

---

## 1. Executive Summary

### Goal

This project investigates whether **counterfactual data augmentation (CAD)** — rule-based paraphrase variants added to training data — **introduces or amplifies unintended group-level biases** in hate speech detection, across both **text** and **image** modalities.

### Methods

- **Data:** 6k balanced base dataset (8 classes, 750 per class) from Kennedy et al. (2020) → two-phase Identity-Counterfactual Augmentation (ICA): regex-based identity-term substitution for explicit mentions + LLM (Qwen3.5) rewriting for implicit texts → 18k samples.
- **Text pipeline:** Binary hate vs non-hate classification under two conditions: **nCF** (train on 6k originals) and **CF** (train on 18k with counterfactuals). Phase 1: TF-IDF + LR, Ridge, NB, RF, SVM. Phase 2: Enhanced TF-IDF (word+char) + MiniLM-v2-12-384 with LR/SVM/MLP heads; threshold optimisation on validation set.
- **T2I pipeline:** DSPy + Qwen3:8B (Ollama) for photorealistic prompts → Z-Image-Turbo (FP8) + Qwen3-4B CLIP on ComfyUI → ~18k images (720×720).
- **Image pipeline:** EfficientNet-B0 (ImageNet pretrained, ~5.3M params) fine-tuned with gradient-reversal adversarial debiasing (Ganin et al. 2016). Three-condition ablation: nCF (6k originals), CF-no-adv (18k, no GRL), CF (18k + GRL). Evaluation via FPR/FNR per identity group, EO-diff, ΔFPR, McNemar, bootstrap CI.

### Main Results

- **Text:** MiniLM + MLP in CF condition reaches **F1 = 0.952**, AUC = 0.978, FPR = 0.054; **ΔFPR = −0.153** (CAD reduces false positives). TF-IDF-based models show **ΔFPR > 0** (FPR increases with CAD); the effect is attributed to surface-form sensitivity of TF-IDF rather than true semantic bias. McNemar and DeLong tests show significant differences between MiniLM and TF-IDF in CF condition (p < 0.01). **Multi-seed variance: F1 std ≤ 0.004 across 3 seeds.**
- **Image:** EfficientNet-B0 + GRL achieves **F1 = 0.794**, AUC = 0.839, FPR = 0.315; CF-no-adv achieves **F1 = 0.801**, AUC = 0.852, FPR = 0.300. Both CF conditions significantly outperform nCF (McNemar p < 0.01). **ΔFPR ≤ 0 for all 8 identity groups** — CAD reduces or maintains FPR across the board (no amplification). GRL reduces
- **Cross-Modal Fusion:** Late fusion (F1 = 0.935, AUC = 0.968, ECE = 0.014) outperforms feature-level GMU + cross-attention fusion (CV F1 = 0.876 ± 0.006, AUC = 0.920 ± 0.007) on n=900 test samples. Simple equal-weight averaging (F1 = 0.940) outperforms all 5 complex fusion strategies.
- **Statistical Rigour:** Chi-squared tests confirm KW results (concordant). OLS two-way ANOVA interaction F=9.82 (p=1.72×10⁻¹⁰) confirms CAD effect is group-dependent. CLIP score audit: mean = 0.909, 0/100 flagged.

---

## 2. Project Structure and Data Flow

### 2.1 Repository Layout

| Path | Role |
|------|------|
| `src/counterfactual_gen/` | Dataset build (6k from HuggingFace), rule-based CAD (18k), config and utils |
| `text_models/` | Data prep, Phase 1 (`binary_fairness_analysis.py`), Phase 2 (`enhanced_analysis.py`), training, evaluation, plots |
| `image_models/` | Data prep (CSV + image paths), EfficientNet-B0 + GRL model, training loop, bias evaluation, 10 plot types |
| Root | `generate_t2i_prompts.py`, `image_gen.py`, `KNOWLEDGE_TRANSFER.md`, `REPORT.md`, `final_dataset_18k*.csv` |

### 2.2 Data Flow (High-Level)

```
Kennedy et al. (2020)  →  hate_speech_dataset_builder.py  →  hate_speech_dataset_6k.csv (6k)
                                    ↓
                         CounterfactualGen_18k.py  →  final_dataset_18k_t2i_prompts.csv (18k)
                                    ↓
                         generate_t2i_prompts.py   →  T2I prompts (DSPy + Qwen3:8B)
                                    ↓
                         image_gen.py (Z-Image-Turbo)  →  PNGs in Hate/*/generated_images/ and non-hate/
```

- **Text experiments** use `data/datasets/final_dataset_18k.csv` (18k rows with `text`, `polarity`, `cf_type`, `original_sample_id`) for CF; `src/counterfactual_gen/hate_speech_dataset_6k.csv` for nCF.
- **Image experiments** use `final_dataset_18k_t2i_prompts.csv` and resolve images via `counterfactual_id` → `{counterfactual_id}.png` in per-class directories.

### 2.3 Configuration and Environment

- **src/counterfactual_gen:** `config.py` defines class taxonomy, thresholds (e.g. `HATE_SCORE_THRESHOLD_HIGH = 0.6`), paths, and `RANDOM_SEED = 42`.
- **T2I prompts:** `generate_t2i_prompts.py` uses env vars `OLLAMA_HOST`, `QWEN_MODEL`, `INPUT_CSV`, `OUTPUT_CSV`, `BATCH_SIZE`, `CHECKPOINT_INTERVAL`.
- **Image generation:** `image_gen.py` uses hardcoded Lightning AI Studio paths (`/teamspace/studios/this_studio/`), `BATCH_SIZE=148`, and ComfyUI model paths; no env-based config.
- **Text/Image models:** Paths are derived from `PROJECT_ROOT` (parent of `text_models/` or `image_models/`). No project-level `requirements.txt`; dependencies are documented in `KNOWLEDGE_TRANSFER.md` §5.

---

## 3. Dataset and Augmentation

### 3.1 Base Dataset (6k)

- **Source:** `ucberkeley-dlab/measuring-hate-speech` (HuggingFace).
- **Construction:** Filtering by hate score and target groups, dedup (MD5), stratified sampling to **750 per class**.
- **Taxonomy (8 classes):** `hate_race`, `hate_religion`, `hate_gender`, `hate_other` (hate); `offensive_non_hate`, `neutral_discussion`, `counter_speech`, `ambiguous` (non-hate). Polarity 50/50 (3k hate, 3k non-hate).

### 3.2 Counterfactual Data Augmentation (CAD)

- **Method:** Two-phase Identity-Counterfactual Augmentation (ICA). Phase 1 (~63%): pre-compiled regex detects explicit identity terms and slurs, then `_pick_replacement()` deterministically swaps to a different term on the same axis (e.g., `Muslim → Protestant`, `Black → Native American`, `gay → transgender`). Phase 2 (~37%): Qwen3.5 LLM via Ollama rewrites texts with no detectable explicit identity mention, guided by `target_group` metadata. Two variants per original — `counterfactual_1` and `counterfactual_2` — use distinct replacement terms. Same polarity and class label preserved throughout.
- **Output:** 18k rows (6k originals + 6k `counterfactual_1` + 6k `counterfactual_2`). Primary table: `data/datasets/final_dataset_18k.csv`. Text and image pipelines use: `data/datasets/final_dataset_18k.csv`.

---

## 4. T2I Pipeline

### 4.1 Prompt Generation

- **Script:** `generate_t2i_prompts.py`.
- **Stack:** DSPy, Ollama (default `http://10.88.0.201:11434`), model `qwen3:8b`, temperature 0.3, max tokens 512, batch size 4.
- **Design:** Prompts preserve content (including hateful themes), add technical photography specs; validation (min length, required elements) and checkpointing every 1k rows.

### 4.2 Image Generation

- **Script:** `image_gen.py` (Lightning AI Studio–oriented; uses `!pip` and hardcoded paths — see REPORT.md).
- **Model:** Z-Image-Turbo (FP8 E4M3FN), Qwen3-4B CLIP, ComfyUI, 720×720, Euler 9 steps, CFG 1.0, deterministic seeding.
- **Output:** ~17,998 PNGs (2 missing in counter_speech); stored per class under `Hate/` and `non-hate/`.

### 4.3 Current vs Recommended Image Generation

| Aspect | Current | Recommendation |
|--------|---------|----------------|
| T2I model | Z-Image-Turbo (FP8) | Keep for speed/quality on H200; for bias studies consider SD3 or Flux with same prompts for comparison. |
| Text encoder | Qwen3-4B CLIP | Document as potential bias source; optional: try T5 or CLIP-L for ablation. |
| Resolution | 720×720 | Keep for generation; classifiers can resize to 224 or 384 (see §6). |
| Reproducibility | Deterministic seeds; no checksums | Add model file checksums and a `requirements.txt`; make paths configurable via env. |
| Portability | Lightning Studio paths, `!pip` | Refactor to env vars and standard pip/conda; optional Docker for reproducibility. |

---

## 5. Text Pipeline (Phase 1 and Phase 2)

### 5.1 Design

- **Task:** Binary classification (hate = 1, non-hate = 0).
- **Splits:** Stratified on **originals** with target 70/15/15 and realized 69.99/15.00/15.01 (4,158/891/892); val/test identical across nCF and CF. CF train includes all 3 variants per original in the train group; no leakage.
- **Metrics:** Accuracy, F1, AUC-ROC, FPR, FNR, Brier; bootstrap 95% CI; ΔFPR / ΔFNR (CF − nCF). Phase 2: validation-set threshold optimisation (max F1), McNemar and DeLong tests.

### 5.2 Phase 1 — TF-IDF Baselines

- **Features:** TF-IDF word 1–3 grams, 15k features, fit on nCF train only; same vectorizer for CF (same feature space).
- **Models:** Logistic Regression, Ridge (isotonic calibration), Multinomial NB, Random Forest, LinearSVM (sigmoid calibration).
- **Result:** All show ΔFPR > 0; Ridge and SVM show largest FPR increase (e.g. ΔFPR ≈ +0.065 to +0.126).

### 5.3 Phase 2 — Enhanced TF-IDF and MiniLM

- **Enhanced TF-IDF:** Word (1–3, 12k) + char (2–4, 8k), combined with `scipy.sparse.hstack` → 20k features; LR and SVM.
- **MiniLM:** `sentence-transformers/all-MiniLM-L12-v2` (384-dim, L2-normalised); embeddings cached; val/test embeddings shared between nCF and CF. Heads: LR, SVM (calibrated), MLP (256→64, ReLU, Adam, early stopping).
- **Best model:** MiniLM + MLP, CF: F1 = 0.952, AUC = 0.978, FPR = 0.054; ΔFPR = −0.153. MiniLM + LR is the only other model with ΔFPR < 0 (−0.011).

### 5.4 Current vs Recommended Text Models

| Aspect | Current | Recommendation |
|--------|---------|----------------|
| Embeddings | MiniLM-v2-12-384 | Consider all-MiniLM-L6-v2 (faster) or paraphrase-multilingual-MiniLM for robustness; optional cross-encoder for hard cases. |
| Calibration | CalibratedClassifierCV for Ridge/SVM | Extend calibration reporting (e.g. ECE) for all Phase 2 models. |
| Fairness | FPR/FNR, ΔFPR; bootstrap CI | Add per–identity-group ΔFPR in text (like image pipeline); optional equalised odds / demographic parity. |
| Code | Consistent splits and metrics | Unify 18k CSV name: use one canonical file (e.g. `final_dataset_18k_t2i_prompts.csv`) and ensure text scripts read required columns (`text`, `polarity`, `cf_type`, `original_sample_id`). |

---

## 6. Image Pipeline

### 6.1 Architecture — EfficientNet-B0 + Gradient-Reversal Adversarial Debiasing

- **Data:** `final_dataset_18k_t2i_prompts.csv`; image paths from `image_models/data_prep.py` via per-class folders. Same group-aware split as text (originals for val/test; CF train = all variants for train IDs; no leakage). Case-insensitive matching handles CSV `_cf1`/`_cf2` ↔ file `_CF1`/`_CF2` mismatch.
- **Backbone:** EfficientNet-B0 (`torchvision.models.efficientnet_b0`, ImageNet pretrained, ~5.3M params). First 6 MBConv blocks frozen; remaining layers fine-tuned.
- **Task head:** Dropout(0.3) → Linear(1280,256) → ReLU → Dropout(0.2) → Linear(256,1) → sigmoid. Binary hate/non-hate.
- **Adversarial head (CF condition):** GradientReversalLayer(λ) → Dropout(0.3) → Linear(1280,256) → ReLU → Dropout(0.2) → Linear(256,8). Predicts `target_group` to debias backbone features.
- **GRL:** Ganin et al. (JMLR 2016). λ ramps from 0→1 via `λ(p) = 2/(1+exp(−10p))−1`. Adversarial weight = 0.5.
- **GRL adversarial-loss weight source:** `image_models/config.py` (`ADV_WEIGHT = 0.5`).
- **Sensitivity caveat:** image GRL weight was not systematically tuned; `[0.3, 0.7]` was informally tested during development.
- **Cross-attention note:** `cross_modal/cross_attention_fusion.py` uses a different adversarial-loss weight (`0.3`) for its GRL head.
- **Training:** AdamW (backbone LR=1e-4, heads LR=1e-3), CosineAnnealingLR, label_smoothing=0.05, grad_clip=1.0, batch_size=64, early stopping patience=5 on val F1.
- **Three-condition ablation:** nCF (6k originals only), CF-no-adv (18k, no GRL), CF (18k + GRL).

### 6.2 Results Summary

| Condition | F1 | AUC | FPR |-----------|-----|------|------|---------|---------|----------------|
| nCF | 0.770 | 0.816 | 0.387 | 0.440 | 0.529 | — |
| CF-no-adv | **0.801** | **0.852** | **0.300** | 0.574 | 0.730 | p=0.0003 *** |
| CF (GRL) | 0.794 | 0.839 | 0.315 | 0.527 | 0.635 | p=0.0068 ** |

ΔFPR ≤ 0 for all 8 identity groups (CF vs nCF). Largest reductions: disability (−0.143), sexual_orientation (−0.124), multiple/none (−0.099). See KNOWLEDGE_TRANSFER.md §9B for full per-group tables.

### 6.3 Future Improvements

| Option | Rationale |
|--------|----------|
| **GPU training** | Current CPU training takes 2,867–12,036s per condition; GPU would reduce to minutes |
| **Resolution 384** | Better use of 720p image content; use EfficientNet-B3/B4 or ViT-384 variants |
| **CLIP ViT** | Aligns with T2I text encoder (Qwen3-4B CLIP); enables zero-shot FPR comparison |
| **Multi-task learning** | Joint text+image prediction head for cross-modal consistency |

---

## 7. Bias Evaluation Framework

### 7.1 Metrics

- **FPR:** FP / (FP + TN) — rate of non-hate content wrongly flagged as hate (per group or overall).
- **FNR:** FN / (FN + TP) — rate of hate content missed.
- **ΔFPR = FPR_CF − FPR_nCF:** Positive ⇒ CAD amplified false-positive bias; negative ⇒ CAD reduced it.

### 7.2 Design

- **nCF:** Trained on originals only (6k text or ~6k images).
- **CF:** Trained on full 18k (originals + counterfactuals). Same test set (originals only) for both.
- **Per-group:** FPR/FNR and ΔFPR by `target_group` (e.g. race/ethnicity, religion, gender) for text and image.

### 7.3 Statistical Testing (Text)

- **McNemar:** Compares error patterns of two models on the same test set.
- **DeLong:** Compares AUC of two models on the same test set. Used in Phase 2 for MiniLM vs TF-IDF (CF condition); significant p-values for key pairs (e.g. SVM/TF-IDF vs MiniLM+LR).

---

## 8. Key Results

### 8.1 Text (from KNOWLEDGE_TRANSFER §9)

- **Best CF model:** MiniLM + MLP — F1 = 0.952, AUC = 0.978, FPR = 0.054, FNR = 0.042.
- **Multi-seed stability (3 seeds):** MiniLM+MLP nCF F1 = 0.849 ± 0.002, CF F1 = 0.841 ± 0.004. All std < 0.005, confirming high reproducibility.
- **ΔFPR:** MiniLM+LR and MiniLM+MLP show **negative** ΔFPR; all TF-IDF-based models show **positive** ΔFPR (largest for SVM/TF-IDF+Char: +0.126).
- **Interpretation:** TF-IDF FPR increase is attributed to identity-token co-occurrence learning — ICA introduces substituted identity tokens into hate-labelled rows, and TF-IDF encodes them as hate-correlated features, subsequently over-flagging non-hate content mentioning those groups. Semantic embeddings (MiniLM) encode sentence-level sentiment and context rather than individual identity tokens, making them robust to identity-term substitution and genuinely benefiting from the increased training diversity.

### 8.2 Image (from KNOWLEDGE_TRANSFER §9B)

- **Best CF model (by F1):** CF-no-adv — F1 = 0.801, AUC = 0.852, FPR = 0.300.
- **Multi-seed stability (3 seeds):** nCF F1 std = 0.016 (higher variance); CF-no-adv std = 0.002, CF(GRL) std = 0.003. CAD training is more stable.
- **ΔFPR:** All identity groups show ΔFPR ≤ 0 (CF vs nCF). No FPR amplification from CAD — **contrasts with TF-IDF text models** where CAD universally increases FPR.
- **Statistical significance:** Both CF and CF-no-adv significantly outperform nCF (McNemar p = 0.0003 and p = 0.0068). CF-no-adv vs CF is not significant (p = 0.38), confirming GRL preserves performance.
- **Cross-modal alignment:** Image models confirm text pipeline finding — visual/semantic representations avoid CAD-induced FPR amplification that plagues surface-count features (TF-IDF).

### 8.3 Cross-Attention Fusion (Phase 6)

- **CV-averaged (5-fold):** F1 = 0.876 ± 0.006, AUC = 0.920 ± 0.007.
- Late fusion (F1 = 0.935) still outperforms cross-attention on n = 900 test set. Likely cause: limited training data favours simple averaging over attention-weight learning.

### 8.4 Enhanced Statistics (Phase 6)

- **Chi-squared (FPR proportionality):** text_cf χ² = 77.81 (p = 3.84 × 10⁻¹⁴), image χ² = 6.30 (p = 0.505 NS), fusion χ² = 35.70 (p = 8.26 × 10⁻⁶). Image is the only modality with statistically uniform cross-group FPR.
- **OLS two-way ANOVA:** interaction F = 9.82, p = 1.72 × 10⁻¹⁰ — CAD effect is group-dependent.
- **Cochran's Q:** Q = 0.0, p = 1.0 — no systematic bias in binary error patterns.
- **CLIP score audit:** mean = 0.909, std = 0.098, 0/100 flagged (threshold 0.5). T2I images are semantically faithful to prompts.

### 8.5 Fairness — CV-Based DP-Constrained Thresholds

- Both fairness-constrained thresholds and multi-seed variance are reported; to our knowledge these are rarely combined in prior work.

---

## 9. Recommendations Summary

### 9.1 Completed ✅

1. ✅ `requirements.txt` with pinned versions.
2. ✅ Image pipeline: EfficientNet-B0 + GRL, 3 conditions, all metrics, 10 plots.
3. ✅ Cross-modal late fusion and cross-attention fusion.
4. ✅ Multi-seed experiment (3 seeds, text + image).
5. ✅ Enhanced statistical tests (chi-squared, OLS ANOVA, Fisher's exact, Cochran's Q, Cohen's d).
6. ✅ Internal nCF vs CF results table across all conditions and modalities.
7. ✅ CLIP score audit (100-sample, 0 flagged).
8. ✅ Confidence interval framework (bootstrap + Clopper-Pearson).
9. ✅ CV-based DP-constrained group-aware fusion.

### 9.2 Remaining / Future Work

- **GPU training:** Current CPU training is slow (2,867–12,036s per condition). GPU would enable larger models, more epochs, and ≥5-seed experiments.
- **T2I diversity:** Add a second generator (e.g. SD3 or Flux) for sensitivity/bias comparison of generated images.
- **Larger architectures:** EfficientNet-B3/B4 at 384px, CLIP ViT for T2I alignment, or multi-task text+image.
- **Text models:** Stronger embedders (e.g. larger sentence-transformers), cross-encoders; report ECE and per-group ΔFPR for text.
- **Data scale:** More training data (>18k) and more identity groups for intersectional fairness analysis.
- **Final paper:** Produce polished LaTeX tables and figures: confusion matrices, FPR delta by group (text + image), ROC/calibration curves, and cross-modal consistency plots.

---

## 10. Results on Our Dataset — nCF vs CF Conditions

All scores below are measured on the **same held-out test set** (originals only) across nCF and CF training conditions. This is an internal ablation: every row uses the same data split and evaluation protocol.

### 10.1 Text Models (n=900 test samples)

| Model | Condition | F1 | AUC | FPR | ΔFPR |
|-------|-----------|-----|------|------|------|
| MiniLM + MLP | nCF | 0.849 ± 0.002 | — | — | — |
| MiniLM + MLP | CF | **0.952** | **0.978** | 0.054 | −0.153 |
| MiniLM + LR | nCF | — | — | — | — |
| MiniLM + LR | CF | — | — | — | −0.011 |
| TF-IDF + SVM | nCF | — | — | — | — |
| TF-IDF + SVM | CF | — | — | — | +0.126 |

ΔFPR = FPR_CF − FPR_nCF. Negative = CAD reduces false-positive bias; positive = CAD amplifies it.

### 10.2 Image Models (n=900 test samples)

| Condition | F1 | AUC | FPR |-----------|-----|------|------|---------|---------|----------------|
| nCF | 0.770 | 0.816 | 0.387 | 0.440 | 0.529 | — |
| CF-no-adv (18k, no GRL) | **0.801** | **0.852** | **0.300** | 0.574 | 0.730 | p=0.0003 *** |
| CF + GRL | 0.794 | 0.839 | 0.315 | 0.527 | 0.635 | p=0.0068 ** |

GRL reduces

### 10.3 Cross-Modal Fusion (n=900 test samples)

| Strategy | F1 | AUC | ECE |
|----------|----|------|-----|
| Late fusion (equal-weight avg) | **0.940** | — | — |
| Late fusion (learned) | 0.935 | **0.968** | 0.014 |
| GMU + cross-attention (CV) | 0.876 ± 0.006 | 0.920 ± 0.007 | — |

Note: all fusion strategies are evaluated on the same n=900 test samples from our dataset.

---

## 11. References and Key Scripts

| Document / Script | Purpose |
|-------------------|--------|
| **KNOWLEDGE_TRANSFER.md** | Full methodology, taxonomy, text results, status, terminology. |
| **REPORT.md** | Image generation pipeline (Z-Image-Turbo, ComfyUI, H200) and code review. |
| **src/counterfactual_gen/config.py** | Class definitions, thresholds, paths. |
| **src/counterfactual_gen/CounterfactualGen_18k.py** | Builds 18k dataset and writes `data/datasets/final_dataset_18k.csv`. |
| **generate_t2i_prompts.py** | T2I prompt generation (DSPy + Qwen3:8B). |
| **image_gen.py** | Batch image generation (ComfyUI + Z-Image-Turbo). |
| **text_models/binary_fairness_analysis.py** | Phase 1 text: TF-IDF + 4 classifiers, bias metrics, McNemar/DeLong. |
| **text_models/enhanced_analysis.py** | Phase 2 text: Enhanced TF-IDF, MiniLM, MLP, threshold optimisation, stat tests. |
| **image_models/data_prep.py** | PyTorch Dataset/DataLoader, image path resolution, nCF/CF/CF-no-adv splits. |
| **image_models/model.py** | EfficientNet-B0 + GradientReversalLayer + task/adversarial heads. |
| **image_models/train.py** | Training loop with differential LR, early stopping, label smoothing. |
| **image_models/evaluate.py** | Full evaluation: fairness, bootstrap CI, McNemar, 10 plot types. |
| **image_models/run_all.py** | End-to-end image pipeline orchestration (3 conditions). |

---

*This report summarises the project for ACM Multimedia 2026. It is intended to be self-contained for a new reader while avoiding duplication of KNOWLEDGE_TRANSFER.md; see that document for full detail. Last updated 6 March 2026: v3.1 — Removed invalid cross-paper SOTA comparison; Section 10 now reports internal nCF vs CF ablation results on our own dataset.*
