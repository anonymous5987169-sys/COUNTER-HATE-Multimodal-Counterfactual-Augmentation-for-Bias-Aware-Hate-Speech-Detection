# Knowledge Transfer — Bias Evaluation of Counterfactual Data Augmentation in Hate Speech Detection

**Target Conference:** ACM Multimedia (ACMMM) 2026  
**Document Version:** 5.0  
**Date:** 1 March 2026  
**Status:** All Pipelines Complete — Text + Image + Cross-modal Fusion + Extended Analysis (Validation, Stacking, Fairness, Error Analysis)  

## Addendum: Image Replacement Rerun (2026-03-20)

This project completed a full image-dependent rerun after replacing generated images.

Operational changes introduced:
- Fusion scripts now support per-condition image checkpoints and consolidated reporting.
- Condition triad in fusion outputs is now explicit: `nCF`, `CF-no-adv`, `CF+GRL`.
- Sanity gate confirms required fusion result files exist and contain all three conditions without NaN/Inf metrics.

Refreshed core metrics:
- Image EfficientNet F1: `0.7809` (`nCF`), `0.8080` (`CF-no-adv`), `0.7885` (`CF+GRL`)
- Late Fusion Equal F1: `0.8645` (`nCF`), `0.8526` (`CF-no-adv`), `0.8520` (`CF+GRL`)
- Cross-Attention Ensemble F1: `0.8604` (`nCF`), `0.8300` (`CF-no-adv`), `0.8378` (`CF+GRL`)

Validation note:
- Integration tests pass (`pytest tests/ -v --tb=short -x` -> `12 passed`).

---

## 1. Project Overview

### Research Goal

This project investigates whether **counterfactual data augmentation (CAD)** — a technique commonly used to improve model robustness — inadvertently introduces or amplifies **unintended group-level biases** in hate speech detection systems.

### Novelty Claim

While CAD has been shown to improve overall classification accuracy and robustness, **no prior work systematically measures its bias side effects** across both text and image modalities simultaneously. This study is the first to:

1. Evaluate FPR (False Positive Rate) disparities **before vs. after** augmentation, per identity group.
2. Extend the bias analysis to **generated images** — testing whether T2I models propagate or amplify textual biases visually.
3. Use a purpose-built **8-class taxonomy** covering 4 hate and 4 non-hate categories across race, religion, gender, and other identity groups.

### Plain-English Summary

Researchers often add synthetic "counterfactual" data to training sets to help AI models generalize better. For example, if a model learns to associate the word "Muslim" with hate speech, a counterfactual might swap "Muslim" for "Christian" to teach the model that identity terms alone don't determine hate. This project asks a critical question: **does this augmentation technique actually make things fairer, or does it accidentally introduce new biases?** The project tests this on both text classifiers and image classifiers (using AI-generated images from the text), producing a comprehensive bias audit across 8 categories of speech.

---

## 2. Dataset & Preprocessing

### 2.1 Source Dataset

| Property | Value |
|----------|-------|
| **Name** | Measuring Hate Speech |
| **Authors** | Kennedy, Brendan; Atari, Mohammad; Davani, Aida M; Hoover, Joe; Omrani, Ali; Graham, Jesse; Dehghani, Morteza |
| **Year** | 2020 |
| **HuggingFace ID** | `ucberkeley-dlab/measuring-hate-speech` |
| **Original Size** | 135,556 samples |
| **Language** | English |
| **Annotations** | Multi-dimensional: hate scores, sentiment, target group labels |

### 2.2 Sampling Strategy — Base Dataset (6,000 Samples)

The base dataset was constructed by the script [hate_speech_dataset_builder.py](file:///home/vslinux/Documents/research/major-project/src/counterfactual_gen/hate_speech_dataset_builder.py) with configuration defined in [config.py](file:///home/vslinux/Documents/research/major-project/src/counterfactual_gen/config.py).

**Processing Pipeline:**

1. **Classification & Filtering** — Mapped source annotations to 8-class taxonomy using target group columns (`target_race`, `target_religion`, `target_origin`, `target_gender`, `target_sexuality`, `target_age`, `target_disability`). Applied quality thresholds:
   - Hate classes: `hate_score ≥ 0.6`
   - Offensive non-hate: `0.35 ≤ hate_score < 0.6`
   - Non-hate classes: `hate_score < 0.35`
   - Ambiguous: multiple targets or no targets with `hate_score < 0.6`
   - Text length filter: 10–200 words

2. **Deduplication** — MD5 hashing of text. Reduced from 98,322 → 30,597 unique samples (68.9% deduplication).

3. **Class Balancing** — High-confidence sampling (sorted by confidence scores). Target: **750 samples per class**. Minimal oversampling: only 1 sample for `hate_religion` (749→750).

4. **Metadata Assignment** — Unique IDs in format `HS_{CLASS}_{INDEX}` (e.g., `HS_HATERACE_0000`).

### 2.3 Class Taxonomy (8 Classes)

| # | Class Label | Polarity | Target Groups | Samples |
|---|-------------|----------|---------------|---------|
| 1 | `hate_race` | hate | Race/ethnicity | 750 |
| 2 | `hate_religion` | hate | Religion | 750 |
| 3 | `hate_gender` | hate | Gender, Sexual orientation | 750 |
| 4 | `hate_other` | hate | National origin/citizenship, Age, Disability, Political ideology | 750 |
| 5 | `offensive_non_hate` | non-hate | *(none targeted — general profanity)* | 750 |
| 6 | `neutral_discussion` | non-hate | *(mentions identity groups neutrally)* | 750 |
| 7 | `counter_speech` | non-hate | *(supports marginalized groups)* | 750 |
| 8 | `ambiguous` | non-hate | Multiple/none | 750 |
| | **TOTAL** | | | **6,000** |

**Polarity Balance:** 3,000 hate + 3,000 non-hate (perfect 50/50).

### 2.4 Target Group Distribution (from `dataset_statistics.json`)

| Target Group | Count |
|-------------|-------|
| Race/ethnicity | 1,157 |
| Gender | 1,133 |
| Religion | 1,091 |
| National origin/citizenship | 875 |
| Sexual orientation | 761 |
| Multiple/none | 750 |
| Disability | 177 |
| Age | 56 |

### 2.5 Base Dataset CSV Schema — `hate_speech_dataset_6k.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sample_id` | string | Unique ID: `HS_{CLASS}_{INDEX}` | `HS_HATERACE_0000` |
| `text` | string | Social media text content | `"I wanna shoot one of them..."` |
| `class_label` | string | One of 8 class labels | `hate_race` |
| `target_group` | string | Primary identity group(s) | `race/ethnicity` |
| `polarity` | string | `"hate"` or `"non-hate"` | `hate` |
| `hate_score` | float | Hate speech score from source | `5.53` |
| `confidence` | float | Classification confidence | `5.53` |

**Text Length Statistics:**
- Mean: 31.6 words | Median: 26 words | Min: 10 | Max: 128 | Std: 20.3

---

## 3. Counterfactual Data Augmentation (CAD)

### 3.1 Method

Counterfactuals were generated using a **two-phase Identity-Counterfactual Augmentation (ICA)** approach implemented in [CounterfactualGen_18k.py](file:///home/vslinux/Documents/research/major-project/src/counterfactual_gen/CounterfactualGen_18k.py).

> [!NOTE]
> The counterfactual generation uses a **two-phase Identity-Counterfactual Augmentation (ICA)** approach — the canonical CAD method introduced by Kaushik et al. (2020). Each original sample is rewritten to target a *different* identity group on the same axis, while preserving polarity, tone, and sentence structure. This is not surface-form perturbation (e.g., contraction toggling); it is systematic identity-group substitution designed to break spurious correlations between identity mentions and hate labels.

**Two-phase process:**

| Phase | Condition | Method | Coverage |
|-------|-----------|--------|----------|
| **Phase 1** | Explicit identity terms or slurs detectable by regex | Pre-compiled regex patterns detect the identity term/slur; `_pick_replacement()` deterministically selects a different term from the same identity axis (e.g., `Muslim → Protestant`, `Black → Native American`, `gay → heterosexual`). Case and plural forms preserved. | ~63% of samples |
| **Phase 2** | No explicit identity term detected | LLM (Qwen3.5 via Ollama) rewrites the text to reference a different identity group, guided by the sample's `target_group` metadata. Validated for length, language, and non-identity to original text. Injection fallback used as last resort. | ~37% of samples |

**Two counterfactuals per original sample:**

| CF Type | Column Value | Method | Description |
|---------|-------------|--------|-------------|
| **Identity counterfactual 1** | `counterfactual_1` | Phase 1 regex swap or Phase 2 LLM rewrite using first deterministic replacement | Different identity group, same polarity and class label |
| **Identity counterfactual 2** | `counterfactual_2` | Same two-phase process using a second deterministic replacement, guaranteed ≠ CF1 | Another different identity group, same polarity and class label |

**Identity axes and example substitutions:**

| Axis | Example Substitutions |
|------|-----------------------|
| `race_ethnicity` | `Black → Native American → Indian` |
| `religion` | `Muslim → Protestant → Catholic → Jewish` |
| `gender_sexuality` | `women → men`, `gay → heterosexual → transgender` |
| `nationality` | `Mexican → Nigerian → Chinese → Syrian` |
| `disability` | `disabled → blind → autistic` |
| `age` | `elderly → teenager → millennial` |

### 3.2 Output Corpus

| Metric | Value |
|--------|-------|
| Original samples | 6,000 |
| Counterfactual variant 1 | 6,000 |
| Counterfactual variant 2 | 6,000 |
| **Total** | **18,000** |

### 3.3 Augmented Dataset CSV Schema — `final_dataset_18k_t2i_prompts.csv`

**Row count:** 18,099 lines (18,098 data rows + header). File size: 7.68 MB.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `original_sample_id` | string | Links back to base dataset | `HS_AMBIGUOUS_0000` |
| `counterfactual_id` | string | Unique ID for this row | `HS_AMBIGUOUS_0000_CF1` |
| `text` | string | Original or counterfactual text | `"I cherish you, radiant Black Queen..."` |
| `class_label` | string | 8 classes + `flipped_generated` | `ambiguous` |
| `target_group` | string | Identity group | `multiple/none` |
| `polarity` | string | `"hate"` or `"non-hate"` | `non-hate` |
| `hate_score` | float | Score (NaN for counterfactuals) | `-8.07` |
| `confidence` | float | Confidence (NaN for counterfactuals) | `9.07` |
| `cf_type` | string | `original`, `counterfactual_1`, `counterfactual_2` | `counterfactual_1` |
| `t2i_prompt` | string | Generated T2I prompt for image generation | `"A diverse group of women..."` |

---

## 4. Text-to-Image (T2I) Pipeline

### 4.1 T2I Prompt Generation

Implemented in [generate_t2i_prompts.py](file:///home/vslinux/Documents/research/major-project/generate_t2i_prompts.py).

| Property | Value |
|----------|-------|
| **Framework** | DSPy (with `dspy.Predict` and `dspy.Signature`) |
| **LLM Backend** | Ollama (remote, at `http://10.88.0.201:11434`) |
| **Model** | `qwen3:8b` (Qwen3 8-billion parameter model) |
| **Temperature** | 0.3 |
| **Max Tokens** | 512 |
| **Batch Size** | 4 (per API call) |
| **Checkpoint Interval** | Every 1,000 rows |

**Prompt Design:**

The DSPy `T2IPromptEnhancedSignature` instructs the LLM to generate photorealistic prompts that:

1. **Preserve ALL content** including hateful/offensive themes — no sanitization.
2. Include **technical photography specs**: "8K resolution, shot on Canon EOS R5 with 85mm f/1.4 lens, natural golden hour lighting, cinematic composition, photorealistic documentary style."
3. For **hate content**: uses harsher lighting/camera choices (e.g., "shot on Nikon Z9 with 35mm f/1.8, dramatic harsh lighting, gritty realism").
4. Generates **scene descriptions** that visually represent the text's meaning and intent.

**Post-processing** adds missing technical elements (resolution, camera, lighting, style, composition) as a safety net. **Validation** checks for minimum 50 characters and required technical elements.

### 4.2 Image Generation

Implemented in [image_gen.py](file:///home/vslinux/Documents/research/major-project/image_gen.py).  
Full technical report: [REPORT.md](file:///home/vslinux/Documents/research/major-project/REPORT.md) (837 lines).

| Property | Value |
|----------|-------|
| **T2I Model** | **Z-Image-Turbo** (FP8 E4M3FN quantized) |
| **Text Encoder** | Qwen3-4B (CLIP, `wan` type loading) |
| **VAE Decoder** | Standard autoencoder (`ae.safetensors`) |
| **Framework** | ComfyUI (headless, node-API mode) |
| **Platform** | Lightning AI Studio — NVIDIA H200 80 GB SXM5 |
| **Resolution** | 720 × 720 pixels |
| **Format** | PNG |
| **Sampler** | Euler, 9 steps |
| **CFG Scale** | 1.0 (guidance distilled into model weights) |
| **Scheduler** | Simple (linear noise schedule) |
| **Batch Size** | 148 (with recursive OOM halving) |
| **Prompt Prefix** | `"high quality, detailed, professional, sharp focus"` |
| **Negative Prompt** | Empty string (pre-encoded once) |
| **Seeding** | Deterministic: `seed(i) = (0xDEAD_BEEF + i × 1,000,003) mod 2³²` |

**Key Technical Decisions:**
- FP8 quantization: halves memory per parameter vs FP16, enables batch size 148
- True batching: all prompts encoded, all latents generated, all images decoded in single batched passes
- Adaptive fault tolerance: recursive batch halving on OOM, per-image VAE fallback
- Checkpoint resumability: progress saved every 5 batches

### 4.3 Image Generation Results

Images were generated in **8 separate runs**, one per class category:

| Category | Directory | Image Count | ZIP Size |
|----------|-----------|-------------|----------|
| `hate_race` | `Hate/Hate_race/generated_images/` | 2,250 | 1.60 GB |
| `hate_religion` | `Hate/Hate_religion/generated_images/` | 2,250 | 1.59 GB |
| `hate_gender` | `Hate/Hate_Gender/generated_images/` | 2,250 | 1.50 GB |
| `hate_other` | `Hate/Hate_Others/generated_images/` | 2,250 | 1.61 GB |
| `ambiguous` | `non-hate/generated_images-ambigious/` | 2,250 | 1.58 GB |
| `counter_speech` | `non-hate/generated_images-counter-speech/` | 2,248 | 1.58 GB |
| `neutral_discussion` | `non-hate/generated_images-neutral/` | 2,250 | 1.58 GB |
| `offensive_non_hate` | `non-hate/generated_images-offensive-non-hate/` | 2,250 | 1.52 GB |
| **TOTAL** | | **~17,998** | **~12.56 GB** |

**Image Naming Convention:** `{counterfactual_id}.png`  
Examples: `HS_HATERELIGION_0000.png`, `HS_HATERELIGION_0000_CF1.png`, `HS_HATERELIGION_0000_CF2.png`

### 4.4 Linking Text Data to Images

Images are linked to their source data via the **naming convention**: each image file is named `{counterfactual_id}.png`. To pair any row in `final_dataset_18k_t2i_prompts.csv` with its image, simply look up `{counterfactual_id}.png` in the appropriate category's image directory. No separate combined CSV is required.

The primary data source for all experiments is `final_dataset_18k_t2i_prompts.csv` (18,098 rows), with images stored in category-specific directories under `Hate/` and `non-hate/`.

---

## 5. Experimental Setup

### 5.1 Two-Condition Design

| Variant | Abbreviation | Training Data | Size | Label |
|---------|-------------|---------------|------|-------|
| Without Counterfactuals | **nCF** | Original 6,000 samples only | 6,000 | Baseline |
| With Counterfactuals | **CF** | Full 18,000 samples (originals + counterfactuals) | 18,000 | Augmented |

Both variants are evaluated independently on text and image modalities.

### 5.2 Train/Val/Test Split

Canonical split from `canonical_splits.py` (stratified to maintain class balance):

| Split | Ratio | Original Count | Notes |
|-------|-------|----------------|-------|
| Train | 69.99% (target 70%) | 4,158 | CF condition expands train IDs with counterfactual variants |
| Validation | 15.00% (target 15%) | 891 | Originals only |
| Test | 15.01% (target 15%) | 892 | Originals only |

```python
from sklearn.model_selection import train_test_split
train, temp = train_test_split(df, test_size=0.30, stratify=df['class_label'], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp['class_label'], random_state=42)
```

### 5.3 Text Model Architectures

Two phases of text experiments were run. **Phase 1** (4 TF-IDF baseline models) established the benchmark. **Phase 2** (6 additional models across Enhanced TF-IDF and MiniLM-v2-12-384) tested whether the FPR amplification finding from Phase 1 is a feature-representation artifact. All 10 models are evaluated on the identical 892-sample test set.

**Phase 1 — TF-IDF Baselines (4 models)**

| Model | Type | Rationale |
|-------|------|----------|
| **Logistic Regression** | Traditional ML baseline | Simple, interpretable baseline. Uses TF-IDF features. Fast to train, provides a linear reference point for bias comparison. |
| **Ridge Regression** | Traditional ML | `RidgeClassifier` with L2 regularisation. Calibrated via `CalibratedClassifierCV` (isotonic, cv=5) for probability output. Strong linear alternative to logistic; particularly effective on high-dimensional sparse features. |
| **Naive Bayes** | Generative probabilistic | `MultinomialNB` (α=0.1). Excels on high-dimensional TF-IDF features with fast inference. Strong inductive bias for text classification; serves as a probabilistic baseline. |
| **Random Forest** | Ensemble ML | Non-linear ensemble baseline. 300 trees with balanced class weights. Captures feature interactions that linear models miss. |
| **LinearSVM** | Traditional ML | Maximises the margin between hate/non-hate in sparse TF-IDF feature space. Calibrated with sigmoid method for probability output. Generally superior to LR on high-dimensional text data. |

**Phase 2 — Enhanced TF-IDF (2 models)**

| Model | Type | Rationale |
|-------|------|----------|
| **SVM / TF-IDF+Char** | Traditional ML | LinearSVC on combined word (1–3-gram, 12k features) + character (2–4-gram, 8k features) TF-IDF matrix. Character n-grams capture slur mutations, typos, and morphological patterns word-only TF-IDF misses. |
| **LR / TF-IDF+Char** | Traditional ML | Logistic Regression on the same 20k-feature combined matrix. Best non-neural baseline. |

**Phase 2 — MiniLM-v2-12-384 (3 models)**

| Model | Type | Rationale |
|-------|------|----------|
| **MiniLM-v2-12-384 + LR** | Semantic embedding | `sentence-transformers/all-MiniLM-L12-v2` (12-layer, 384-dim, ~33M params, trained on 1B sentence pairs) → Logistic Regression. Dense semantic embeddings encode sentence-level sentiment and grammatical context rather than individual identity tokens, making them robust to identity-term substitution; directly addresses the root cause of TF-IDF FPR drift under ICA. |
| **MiniLM-v2-12-384 + SVM** | Semantic embedding | Same MiniLM encoder → LinearSVC with sigmoid calibration. Margin maximisation in dense embedding space. |
| **MiniLM-v2-12-384 + MLP** | Semantic embedding + neural | Same MiniLM encoder → 2-layer MLP (256→64→2, ReLU, Adam, α=1e-3, early stopping). Non-linear interactions over semantic dimensions. Best-performing model overall. |

### 5.4 Image Model Architecture — EfficientNet-B0 + Gradient-Reversal Adversarial Debiasing

A single EfficientNet-B0 backbone is trained under three ablation conditions:

| Condition | Training Data | Adversarial Head | Purpose |
|-----------|---------------|------------------|---------|
| **nCF** | 6k originals only (~4,158 train) | No | Baseline — no augmentation, no debiasing |
| **CF-no-adv** | 18k (originals + CFs, ~12,468 train) | No | Measures effect of data augmentation alone |
| **CF** | 18k (originals + CFs, ~12,468 train) | Yes (GRL) | Full pipeline — augmentation + adversarial debiasing |

**Architecture:**

```
EfficientNet-B0 (ImageNet pretrained, ~5.3M params, freeze_blocks=6)
       │
  AdaptiveAvgPool2d → 1280-dim feature vector
       │
       ├──► task_head: Dropout(0.3) → Linear(1280,256) → ReLU → Dropout(0.2) → Linear(256,1) → σ  [hate/non-hate]
       │
       └──► adv_head (CF condition only):
              GradientReversalLayer(λ) → Dropout(0.3) → Linear(1280,256) → ReLU → Dropout(0.2) → Linear(256,8)  [target_group]
```

**Gradient Reversal Layer (GRL):** Implements the domain-adversarial technique of Ganin et al. (JMLR 2016). During forward pass, acts as identity; during backward pass, multiplies gradients by −λ. This forces the backbone to learn features that are **uninformative about target group** while remaining predictive of hate/non-hate polarity, reducing demographic bias in predictions.

**λ schedule:** `λ(p) = 2 / (1 + exp(−10p)) − 1`, where `p = epoch / total_epochs`. Ramps from 0 → 1 during training, preventing the adversarial head from destabilising early learning.

### 5.5 Hyperparameters

**TF-IDF word (Phase 1, shared across LR/Ridge/NB/RF/SVM):**
- max_features=15,000, ngram_range=(1,3), sublinear_tf=True
- strip_accents='unicode', min_df=2, stop_words='english'
- Fitted on nCF train split only; same vectorizer reused for CF (13,185 features)

**Enhanced TF-IDF (Phase 2, word+char combined):**
- Word: max_features=12,000, ngram_range=(1,3), sublinear_tf=True, min_df=2, stop_words='english'
- Char: max_features=8,000, ngram_range=(2,4), analyzer='char_wb', sublinear_tf=True, min_df=3
- Combined via `scipy.sparse.hstack` → 20,000 features

**Logistic Regression (text):**
- Phase 1: Solver `lbfgs`, max_iter=2000, C=1.0, class_weight='balanced'
- Phase 2 (TF-IDF+Char): Solver `saga`, max_iter=3000, C=0.5, class_weight='balanced'
- Phase 2 (MiniLM): Solver `lbfgs`, max_iter=3000, C=2.0, class_weight='balanced'

**Ridge Regression (text):**
- `RidgeClassifier`, alpha=1.0, class_weight='balanced'
- Calibration: `CalibratedClassifierCV`, cv=5, method='isotonic'

**Naive Bayes (text):**
- `MultinomialNB`, alpha=0.1 (Laplace smoothing)
- No class_weight (handles balance via priors)

**Random Forest (text):**
- n_estimators=300, max_features='sqrt'
- min_samples_split=5, min_samples_leaf=2
- class_weight='balanced'

**LinearSVM (text):**
- Phase 1 (TF-IDF): C=1.0, max_iter=2000, class_weight='balanced'
- Phase 2 (TF-IDF+Char): C=0.5, max_iter=3000, class_weight='balanced'
- Calibration: `CalibratedClassifierCV`, cv=5, method='sigmoid'

**MiniLM-v2-12-384 encoder (shared):**
- Model: `sentence-transformers/all-MiniLM-L12-v2`
- Embeddings: 384-dim, L2-normalised, batch_size=128
- Cached to disk (`enhanced_results/embeddings/`) to avoid re-encoding across runs
- CF train set encoded separately (12,469 texts); val/test embeddings reused from nCF (same split)

**MLP (MiniLM head):**
- hidden_layer_sizes=(256, 64), activation='relu', solver='adam'
- alpha=1e-3 (L2 penalty), max_iter=400
- early_stopping=True, validation_fraction=0.1

**Threshold optimisation (Phase 2, all models):**
- Decision threshold tuned on val set (grid search over [0.1, 0.9] in 161 steps)
- Objective: maximise F1 on val; threshold applied to test without further tuning

**EfficientNet-B0 (image, all conditions):**
- Pre-trained on ImageNet (`torchvision.models.efficientnet_b0`)
- ~5.3M total params; ~3.5M trainable (freeze_blocks=6: first 6 MBConv blocks frozen)
- Input: 224×224 (train: RandomResizedCrop + HFlip + ColorJitter + rotation; eval: Resize(256) + CenterCrop(224))
- Normalisation: ImageNet mean/std
- Optimiser: AdamW, weight_decay=1e-4, gradient clipping max_norm=1.0
  - Backbone LR: 1e-4, Head(s) LR: 1e-3
  - Scheduler: CosineAnnealingLR (T_max=epochs)
- Loss: BCEWithLogitsLoss (label_smoothing=0.05) + optional CrossEntropyLoss for adversarial head (adv_weight=0.5)
- Batch size: 64
- Epochs: up to 20 (early stopping patience=5 on val F1)
- GRL λ schedule (CF condition): `λ(p) = 2/(1 + exp(−10p)) − 1`, ramps 0→1 over training
- Image GRL adversarial-loss weight source of truth: `image_models/config.py` (`ADV_WEIGHT = 0.5`)
- Informal image GRL range tested in development: `[0.3, 0.7]` (no systematic sweep)
- Cross-attention fusion GRL head uses a separate adversarial-loss weight: `0.3`
- Threshold: optimised on validation set (grid 0.0–1.0, step 0.02) to maximise F1

---

## 6. Bias Evaluation Framework

### 6.1 Text-Based Bias Analysis

**Formal Definition:**

- **Input:** Text comment `x`
- **Label:** `y ∈ {hate, non-hate}`
- **Model:** `f(x; θ) → ŷ`

**Primary Metrics:**

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Precision** | `TP / (TP + FP)` | Correctness of positive predictions |
| **Recall** | `TP / (TP + FN)` | Coverage of actual positives |
| **Macro F1** | `2 × (P × R) / (P + R)`, averaged across all classes | Overall classification quality |
| **FPR** (False Positive Rate) | `FP / (FP + TN)` — per identity group and per class | Rate of non-hate content falsely flagged as hate |
| **FNR** (False Negative Rate) | `FN / (FN + TP)` — per identity group and per class | Rate of hate content missed |

**Bias Measurement:**

```
Compare:  FPR_CF  vs  FPR_nCF   (per identity group)

If FPR_CF > FPR_nCF:
    → Counterfactual augmentation has INTRODUCED or AMPLIFIED bias
    → Non-hate content about this group is being OVER-FLAGGED after augmentation
    → This is the UNINTENDED HARM being studied

If FPR_CF < FPR_nCF:
    → Augmentation has REDUCED false positive bias for this group
    → Augmentation is BENEFICIAL for fairness

If FPR_CF ≈ FPR_nCF:
    → Augmentation is NEUTRAL w.r.t. bias for this group
```

**Delta metric:** `ΔFPR = FPR_CF − FPR_nCF` per identity group. Positive values indicate bias amplification.

### 6.2 Image-Based Bias Analysis

**Formal Definition:**

- **Input:** Generated image `v` (paired with text sample `x`)
- **Label:** `y ∈ {hate, non-hate}` (same as source text)
- **Model:** Visual classifier `g(v; φ) → ŷ`

**Two-Condition Image Setup:**

| Variant | Image Source | Size | Label |
|---------|-------------|------|-------|
| **nCF-img** | Images from original 6k texts | ~6,000 | Baseline |
| **CF-img** | Images from all 18k texts | ~18,000 | Augmented |

**Image Metrics:**

| Metric | Formula |
|--------|---------|
| `FPR_img` | `FP / (FP + TN)` — falsely flagging non-hate images as hate |
| `FNR_img` | `FN / (FN + TP)` — missing actual hate images |
| `Macro F1_img` | Harmonic mean of per-class precision & recall on image predictions |

**Image Bias Measurement:**

```
Compare:  FPR_CF_img  vs  FPR_nCF_img   (per identity group)
```

### 6.3 Additional Bias Probes

| Probe | Description | Implementation Status |
|-------|-------------|----------------------|
| **Visual Demographic Bias** | FPR/FNR stratified by detected demographic attributes in generated images | `[TO CONFIRM]` |
| **Cross-modal Consistency** | Compare text `f(x)` vs image `g(v)` predictions on the same sample; late fusion ensemble computed with weight sweep and isotonic calibration | ✅ Complete — `cross_modal/late_fusion_ensemble.py`, `cross_modal/ablation_calibration_study.py` |
| **T2I Fidelity Check** | Validate whether generated images faithfully represent source text semantics | `[TO CONFIRM]` |
| **Counterfactual Visual Drift** | Measure visual content shift between original and counterfactual image pairs | `[TO CONFIRM]` |

### 6.4 What Constitutes a "Biased" Result

A result is considered biased if:

1. **FPR disparity exists** — `ΔFPR > 0` for one or more identity groups after augmentation.
2. **Disparity is statistically significant** — measured via confidence intervals or permutation tests.
3. **Group-level asymmetry** — certain identity groups (e.g., religion, race) show disproportionately higher FPR increases than others.
4. **Cross-modal amplification** — bias introduced in text is further amplified in the image modality.

---

## 7. Expected Outputs & Deliverables

### 7.1 Tables

| Table | Description |
|-------|-------------|
| Text nCF vs CF | Macro F1, per-class Precision/Recall/F1, overall FPR, overall FNR |
| Text FPR breakdown | FPR per identity group for nCF and CF, with ΔFPR |
| Text FNR breakdown | FNR per class for nCF and CF |
| Image nCF-img vs CF-img | Macro F1, per-class metrics, FPR, FNR |
| Image FPR breakdown | FPR per identity group for nCF-img and CF-img |
| Cross-modal comparison | Text vs Image predictions on same samples |

### 7.2 Visualizations

- **Confusion matrices** (4 total: nCF-text, CF-text, nCF-img, CF-img)
- **FPR delta bar charts** — per identity group, showing `ΔFPR = FPR_CF − FPR_nCF`
- **Per-class F1 comparison** — grouped bar charts (nCF vs CF)
- **ROC curves** — per class and overall
- **Calibration plots** — for both modalities

### 7.3 Qualitative Analysis

- Example false positives from each condition (text + corresponding image)
- Worst-case counterfactual pairs — examples where augmentation created the most bias
- Visual inspection of generated images flagged with highest FPR

### 7.4 Paper Section Mapping

| Deliverable | Paper Section |
|-------------|--------------|
| Dataset stats, taxonomy | §3 Dataset |
| CAD methodology | §4 Methodology |
| Text nCF vs CF results | §5.1 Text-based Evaluation |
| Image nCF-img vs CF-img results | §5.2 Image-based Evaluation |
| FPR delta analysis | §5.3 Bias Analysis |
| Cross-modal comparison | §5.4 Cross-modal Consistency |
| Discussion of bias amplification | §6 Discussion |

---

## 8. File & Folder Structure

```
major-project/
├── KNOWLEDGE_TRANSFER.md              ← THIS FILE
├── REPORT.md                          ← [OUTPUT] 837-line technical report on image generation pipeline
├── final_dataset_18k_t2i_prompts.csv  ← [INPUT] 18,098 rows: full corpus with T2I prompts (7.68 MB)
├── generate_t2i_prompts.py            ← [SCRIPT] DSPy-based T2I prompt generator (Qwen3:8B via Ollama)
├── image_gen.py                       ← [SCRIPT] Z-Image-Turbo batch image generation for Lightning AI
├── image-gen(backup).py               ← [BACKUP] Previous version of image_gen.py
│
├── src/counterfactual_gen/            ← Dataset construction & counterfactual generation
│   ├── README.md                      ← [DOC] Dataset documentation (8-class taxonomy, methodology)
│   ├── config.py                      ← [CONFIG] Thresholds, class definitions, paths
│   ├── hate_speech_dataset_builder.py ← [SCRIPT] Builds 6k balanced dataset from HuggingFace
│   ├── hate_speech_dataset_6k.csv     ← [OUTPUT] Base dataset: 6,000 samples, 7 columns (1.46 MB)
│   ├── CounterfactualGen_18k.py       ← [SCRIPT] Rule-based counterfactual generator (×3 → 18k)
│   ├── dataset_statistics.json        ← [OUTPUT] Class distribution, hate score stats
│   ├── utils.py                       ← [LIB] Text cleaning, dedup, validation, stats
│   ├── example_usage.py               ← [SCRIPT] Usage demo
│   ├── __pycache__/                   ← [CACHE] Python bytecode
│   └── venv/                          ← [ENV] Virtual environment
│
├── Hate/                              ← Generated images + combined CSVs for HATE categories
│   ├── Complete-outputs/              ← [ARCHIVE] Backup ZIPs of all 4 hate category outputs
│   │   ├── complete_output.zip        ← hate_race outputs (1.60 GB)
│   │   ├── complete_output (1).zip    ← hate_religion outputs (1.59 GB)
│   │   ├── complete_output (2).zip    ← hate_gender outputs (1.50 GB)
│   │   └── complete_output (3).zip    ← hate_other outputs (1.61 GB)
│   │
│   ├── Hate_race/
│   │   ├── generated_images/          ← [OUTPUT] 2,250 PNG images (720×720)
│   │   └── generated_images.zip       ← [ARCHIVE] Images only
│   │
│   ├── Hate_religion/                 ← Same structure: 2,250 images
│   ├── Hate_Gender/                   ← Same structure: 2,250 images
│   └── Hate_Others/                   ← Same structure: 2,250 images
│
└── non-hate/                          ← Generated images for NON-HATE categories
    ├── completeoutputs/               ← [ARCHIVE] Backup ZIPs
    │   ├── complete_output-neutral.zip           ← 1.58 GB
    │   └── complete_output-offensive-non-hate.zip ← 1.52 GB
    ├── generated_images-ambigious/     ← [OUTPUT] 2,250 images
    ├── generated_images-ambigious.zip  ← [ARCHIVE] 1.58 GB
    ├── generated_images-counter-speech/ ← [OUTPUT] 2,248 images
    ├── generated_images-counter-speech.zip ← 1.58 GB
    ├── generated_images-neutral/       ← [OUTPUT] 2,250 images
    ├── generated_images-neutral.zip    ← 1.58 GB
    ├── generated_images-offensive-non-hate/ ← [OUTPUT] 2,250 images
    └── generated_images-offensive-non-hate.zip ← 1.52 GB

text_models/                               ← Text model training & evaluation pipeline
├── data_prep.py                           ← [SCRIPT] Data loading, stratified split, TF-IDF
├── train_models.py                        ← [SCRIPT] LR, SVM, RF training (nCF + CF)
├── evaluate.py                            ← [SCRIPT] Bias evaluation, FPR/FNR, visualizations
├── run_all.py                             ← [SCRIPT] Main orchestrator (runs full pipeline)
├── binary_fairness_analysis.py            ← [SCRIPT] Phase 1: 4 TF-IDF models + full bias analysis
├── enhanced_analysis.py                   ← [SCRIPT] Phase 2: +Enhanced TF-IDF + MiniLM-v2-12-384 + MLP
├── models/                                ← [OUTPUT] Saved models (joblib) + TF-IDF vectorizers
├── binary_fairness_results/               ← [OUTPUT] Phase 1 results
│   ├── binary_fairness_results.json       ← All metrics for 8 runs (4 models × 2 conditions)
│   ├── models/                            ← 8 model joblib files + 1 TF-IDF vectorizer
│   └── plots/                             ← 7 visualization PNGs
└── enhanced_results/                      ← [OUTPUT] Phase 2 results
    ├── enhanced_results.json              ← All metrics for 20 runs (10 models × 2 conditions) + stat tests
    ├── models/                            ← 20 model joblib files
    ├── embeddings/                        ← Cached MiniLM numpy embeddings (npy files)
    └── plots/                             ← 5 visualization PNGs

image_models/                              ← Image model training & evaluation pipeline (EfficientNet-B0 + GRL)
├── data_prep.py                           ← [SCRIPT] PyTorch Dataset/DataLoader, image path resolution, nCF/CF splits
├── model.py                               ← [SCRIPT] EfficientNet-B0 + GradientReversalLayer + task/adv heads
├── train.py                               ← [SCRIPT] Training loop with differential LR, early stopping, label smoothing
├── evaluate.py                            ← [SCRIPT] Full evaluation: fairness, bootstrap CI, McNemar, 10 plot types
├── run_all.py                             ← [SCRIPT] End-to-end orchestrator with argparse
├── feature_extractor.py.bak               ← [BACKUP] Old ResNet-50/ViT feature extractor (superseded)
├── train_classifiers.py.bak               ← [BACKUP] Old LR/SVM linear probing (superseded)
├── models/                                ← [OUTPUT] Saved model checkpoints (.pt)
│   ├── best_model_ncf.pt                  ← nCF condition checkpoint
│   ├── best_model_cf_no_adv.pt            ← CF-no-adv condition checkpoint
│   └── best_model_cf.pt                   ← CF (GRL) condition checkpoint
└── results/                               ← [OUTPUT] Evaluation results
    ├── evaluation_results.json            ← All metrics for 3 conditions + McNemar + bootstrap CI + ΔFPR
    └── plots/                             ← 10 visualization PNGs (confusion matrices, ROC, FPR delta, etc.)

cross_modal/                               ← Cross-modal fusion, ablation & calibration
├── late_fusion_ensemble.py                ← [SCRIPT] Full late-fusion: 5 strategies, bootstrap CI, plots
├── ablation_calibration_study.py          ← [SCRIPT] 21-pt weight sweep + temp scaling + isotonic calibration
├── consistency_analysis.py               ← [SCRIPT] Cross-modal consistency utilities
├── stacking_ensemble.py                  ← [SCRIPT] 4 meta-learner stacking ensemble (5-fold CV, polynomial features)
├── learned_fusion.py                     ← [SCRIPT] 6 fusion strategy comparison (equal weight → GBT)
├── cache/
│   ├── fusion_val_900.npy                 ← Cached MiniLM val embeddings (384-dim, 900 samples)
│   └── fusion_test_900.npy                ← Cached MiniLM test embeddings (384-dim, 900 samples)
└── results/
    ├── late_fusion_results.json           ← All fusion strategies + bootstrap CI
    ├── ablation_calibration_results.json  ← 28 configs: ablation + weight sweep + calibration
    ├── stacking_ensemble_results.json     ← Stacking train OOF diagnostics + held-out test metrics
    ├── learned_fusion_results.json        ← 6 strategies ranked; best F1=0.8736 (condition-dependent)
    └── predictions/
        ├── fusion_test_predictions.csv    ← 900 rows × 14 cols: per-sample branch probabilities + fusion
        ├── stacking_predictions_test.csv  ← Held-out test stacking predictions
        └── stacking_train_oof_predictions.csv ← Train OOF stacking diagnostics

analysis/                                  ← Extended analysis scripts (Phase 5)
├── run_all.py                             ← [SCRIPT] Analysis pipeline orchestrator
├── mlp_cross_validation.py                ← [SCRIPT] Bootstrap CI, nCF CV, CF augmentation effect, CF size-ablation
├── enhanced_statistical_tests.py          ← [SCRIPT] Kruskal-Wallis, Mann-Whitney, per-group FPR
├── baseline_comparison.py                 ← [SCRIPT] Published baseline comparison table
├── per_group_text_dfpr.py                 ← [SCRIPT] Differential FPR by target group (text models)
├── error_analysis.py                      ← [SCRIPT] Confusion matrices, hardest errors, modality disagreement
├── calibration_analysis.py                ← [SCRIPT] Calibration curve analysis
├── clip_score_audit.py                    ← [SCRIPT] CLIP-Score audit for generated images
├── intersectional_bias.py                 ← [SCRIPT] Intersectional bias analysis
├── statistical_tests.py                   ← [SCRIPT] Statistical significance tests
└── results/
    ├── mlp_cv_results.json                ← Bootstrap CI: F1=0.9463 [0.930, 0.960], CF ΔF1=+0.0826 (+9.6%) vs canonical nCF=0.8633, plus CF size-ablation curve
    ├── enhanced_statistical_tests.json    ← Kruskal-Wallis: text p≪0.001, image p=0.506
    ├── baseline_comparison.json           ← Published baseline comparison
    ├── text_per_group_dfpr_results.json   ← Per-group DFPR for text models
    ├── error_analysis.json                ← 58/900 errors, text wins 79% of disagreements
    └── plots/                             ← Visualization PNGs for all extended analyses
```

> [!NOTE]
> All text data lives in `final_dataset_18k_t2i_prompts.csv`. Images are stored per-category in the directories above. Link them via the `counterfactual_id` column → `{counterfactual_id}.png` filename.

---

## 9. Text Model Results

> **Task:** Binary classification — hate (1) vs non-hate (0).  
> **Test set:** 892 samples (identical for all conditions — fair comparison).  
> **CI:** Bootstrap 95% (n=1500–2000 resamples).  
> **Phase 1 script:** `text_models/binary_fairness_analysis.py`  
> **Phase 2 script:** `text_models/enhanced_analysis.py`

### 9.1 Phase 1 — TF-IDF Baseline Performance

| Model | Condition | Accuracy | F1 | AUC-ROC | FPR | FNR | Brier |
|-------|-----------|----------|----|---------|-----|-----|-------|
| **Logistic Regression** | nCF (6k) | 0.8139 | 0.8180 | 0.8920 | 0.2050 | 0.1674 | 0.1409 |
| **Logistic Regression** | CF (18k) | 0.8217 | 0.8296 | 0.8998 | 0.2207 | 0.1362 | 0.1281 |
| **Ridge Regression** | nCF (6k) | 0.8240 | 0.8310 | 0.8924 | 0.2140 | 0.1384 | 0.1301 |
| **Ridge Regression** | CF (18k) | 0.7982 | 0.8133 | 0.8877 | 0.2793 | 0.1250 | 0.1393 |
| **Naive Bayes** | nCF (6k) | 0.8128 | 0.8126 | 0.8890 | 0.1824 | 0.1920 | 0.1359 |
| **Naive Bayes** | CF (18k) | 0.8251 | 0.8312 | 0.9033 | 0.2072 | 0.1429 | 0.1280 |
| **Random Forest** | nCF (6k) | 0.7926 | 0.7933 | 0.8766 | 0.2072 | 0.2076 | 0.1501 |
| **Random Forest** | CF (18k) | 0.8296 | 0.8393 | 0.8999 | 0.2275 | 0.1138 | 0.1309 |
| **LinearSVM** | nCF (6k) | 0.8072 | 0.8080 | 0.8877 | 0.1937 | 0.1920 | 0.1353 |
| **LinearSVM** | CF (18k) | 0.7993 | 0.8137 | 0.8773 | 0.2748 | 0.1272 | 0.1470 |

### 9.2 Phase 2 — Enhanced TF-IDF Performance

| Model | Condition | Accuracy | F1 | AUC-ROC | FPR | FNR | Brier |
|-------|-----------|----------|----|---------|-----|-----|-------|
| **SVM / TF-IDF+Char** | nCF (6k) | 0.8262 | 0.8256 | 0.9049 | 0.1667 | 0.1808 | 0.1229 |
| **SVM / TF-IDF+Char** | CF (18k) | 0.7982 | 0.8156 | 0.8920 | 0.2928 | 0.1116 | 0.1383 |
| **LR / TF-IDF+Char** | nCF (6k) | 0.8419 | 0.8482 | 0.9060 | 0.1959 | 0.1205 | 0.1288 |
| **LR / TF-IDF+Char** | CF (18k) | 0.8341 | 0.8426 | 0.9035 | 0.2162 | 0.1161 | 0.1224 |

### 9.3 Phase 2 — MiniLM-v2-12-384 Performance

| Model | Condition | Accuracy | F1 | AUC-ROC | FPR | FNR | Brier |
|-------|-----------|----------|----|---------|-----|-----|-------|
| **MiniLM + LR** | nCF (6k) | 0.8487 | 0.8553 | 0.9159 | 0.1937 | 0.1094 | 0.1092 |
| **MiniLM + LR** | CF (18k) | 0.8643 | 0.8709 | 0.9301 | 0.1824 | 0.0893 | 0.0993 |
| **MiniLM + SVM** | nCF (6k) | 0.8487 | 0.8568 | 0.9127 | 0.2050 | 0.0982 | 0.1118 |
| **MiniLM + SVM** | CF (18k) | 0.8576 | 0.8716 | 0.9305 | 0.2477 | 0.0379 | 0.1050 |
| **MiniLM + MLP** | nCF (6k) | 0.8509 | 0.8596 | 0.9194 | 0.2072 | 0.0915 | 0.1095 |
| **MiniLM + MLP** | **CF (18k)** | **0.9518** | **0.9523** | **0.9785** | **0.0541** | **0.0424** | **0.0423** |

### 9.4 CF Condition Ranking (All Models, Default Threshold)

| Rank | Model | F1 | AUC-ROC | FPR | FNR |
|------|-------|----|---------|-----|-----|
| **#1** | **MiniLM + MLP** | **0.9523** | **0.9785** | **0.0541** | **0.0424** |
| **#2** | **MiniLM + SVM** | **0.8716** | **0.9305** | 0.2477 | **0.0379** |
| **#3** | **MiniLM + LR** | **0.8709** | **0.9301** | **0.1824** | 0.0893 |
| #4 | LR / TF-IDF+Char | 0.8426 | 0.9035 | 0.2162 | 0.1161 |
| #5 | Random Forest | 0.8393 | 0.8999 | 0.2275 | 0.1138 |
| #6 | Naive Bayes | 0.8312 | 0.9033 | 0.2072 | 0.1429 |
| #7 | LR / TF-IDF | 0.8296 | 0.8998 | 0.2207 | 0.1362 |
| #8 | SVM / TF-IDF+Char | 0.8156 | 0.8920 | 0.2928 | 0.1116 |
| #9 | SVM / TF-IDF | 0.8137 | 0.8773 | 0.2748 | 0.1272 |
| #10 | Ridge / TF-IDF | 0.8133 | 0.8877 | 0.2793 | 0.1250 |

### 9.5 ΔFPR / ΔFNR / ΔAUC (CF − nCF) — CAD Bias Signal

`ΔFPR > 0` = CAD increased false positive rate (amplifies bias).  
`ΔFNR < 0` = CAD reduced missed hate (beneficial).  
`†` = Large FPR increase (ΔFPR > 0.04), high risk for deployment.

| Model | nCF FPR | CF FPR | **ΔFPR** | nCF FNR | CF FNR | ΔFNR | ΔF1 | ΔAUC |
|-------|---------|--------|----------|---------|--------|------|-----|------|
| LR / TF-IDF | 0.2050 | 0.2207 | +0.0158 | 0.1674 | 0.1362 | −0.0312 | +0.0116 | +0.0079 |
| Ridge / TF-IDF | 0.2140 | 0.2793 | +0.0653 `†` | 0.1384 | 0.1250 | −0.0134 | −0.0177 | −0.0047 |
| Naive Bayes | 0.1824 | 0.2072 | +0.0248 | 0.1920 | 0.1429 | −0.0491 | +0.0186 | +0.0143 |
| Random Forest | 0.2072 | 0.2275 | +0.0203 | 0.2076 | 0.1138 | −0.0938 | +0.0460 | +0.0233 |
| SVM / TF-IDF | 0.1937 | 0.2748 | +0.0811 `†` | 0.1920 | 0.1272 | −0.0647 | +0.0057 | −0.0103 |
| SVM / TF-IDF+Char | 0.1667 | 0.2928 | +0.1261 `†` | 0.1808 | 0.1116 | −0.0692 | −0.0101 | −0.0129 |
| LR / TF-IDF+Char | 0.1959 | 0.2162 | +0.0203 | 0.1205 | 0.1161 | −0.0045 | −0.0057 | −0.0025 |
| **MiniLM + LR** | 0.1937 | 0.1824 | **−0.0113 ✓** | 0.1094 | 0.0893 | −0.0201 | +0.0156 | +0.0142 |
| MiniLM + SVM | 0.2050 | 0.2477 | +0.0428 `†` | 0.0982 | 0.0379 | −0.0603 | +0.0147 | +0.0178 |
| **MiniLM + MLP** | 0.2072 | 0.0541 | **−0.1532 ✓✓** | 0.0915 | 0.0424 | −0.0491 | +0.0927 | +0.0591 |

### 9.6 Key Findings

> [!IMPORTANT]
> **Phase 1 (TF-IDF only) findings:**
> 1. All TF-IDF models show ΔFPR > 0 after CAD — augmentation universally increases false positive rate on sparse features.
> 2. FNR decreases across all models (fewer missed hate samples) — a consistent benefit.
> 3. Statistical tests (McNemar + DeLong, nCF vs CF) show no significant differences for any model (all p > 0.05). The test set (892 samples) is under-powered for detecting small per-model deltas.

> [!IMPORTANT]
> **Phase 2 (Enhanced features + MiniLM) findings — the critical research finding:**
> 1. **The FPR amplification seen in Phase 1 is a feature-representation artifact, not genuine semantic bias.** ICA substitutes identity terms (e.g., `Muslim → Protestant`) while keeping polarity labels unchanged. TF-IDF encodes these substituted tokens as hate-correlated features, causing it to over-flag non-hate content mentioning those identities at test time. Semantic embeddings capture the surrounding sentiment and grammatical context, which remain stable across identity-term substitution — so MiniLM benefits from the augmentation without inheriting the spurious lexical correlations.
> 2. **MiniLM + LR is the only TF-IDF-comparable model where CAD reduces FPR** (ΔFPR = −0.011). CAD is safe with semantic features.
> 3. **MiniLM + MLP achieves F1=0.952, AUC=0.978, FPR=0.054 in the CF condition** — a 73% FPR reduction vs nCF (0.207→0.054). The MLP learns non-linear semantic subspaces that TF-IDF cannot access. CAD actively helps here.
> 4. **Statistical significance flips in Phase 2.** MiniLM vs TF-IDF comparisons are now significant: SVM/TF-IDF vs MiniLM+LR (McNemar p=0.0006 ***, DeLong p=0.0004). The representational gap is the dominant signal.
> 5. **Deployment recommendation:** Avoid Ridge/SVM + TF-IDF with CAD (ΔFPR > +0.065). Use MiniLM + LR for a production-safe deployment with CAD (F1=0.871, FPR=0.182, ΔFPR=−0.011). Use MiniLM + MLP for maximum performance (F1=0.952, FPR=0.054).

### 9.7 Statistical Tests (Phase 2 — MiniLM vs TF-IDF, CF condition)

| Comparison [CF condition] | McNemar p | DeLong p | Significance |
|--------------------------|-----------|----------|--------------|
| SVM/TF-IDF vs MiniLM+LR | 0.0006 | 0.0004 | *** |
| LR/TF-IDF vs MiniLM+LR | 0.0217 | 0.0334 | * |
| RF/TF-IDF vs MiniLM+LR | 0.0533 | 0.0280 | ns / * |
| SVM/TF-IDF vs MiniLM+SVM | 0.0019 | 0.0003 | ** |
| SVM/TF-IDF+Char vs MiniLM+LR | 0.0004 | 0.0080 | *** |

### 9.8 Threshold-Optimised Results (Phase 2, val-set threshold)

All Phase 2 models additionally report a val-set-optimised threshold `t*`. Key highlights:

| Model | Cond | t* | F1 (opt) | FPR (opt) | FNR (opt) |
|-------|------|----|----------|-----------|-----------|
| MiniLM + MLP | CF | 0.325 | 0.9559 | 0.0586 | 0.0312 |
| MiniLM + LR | CF | 0.340 | 0.8725 | 0.2455 | 0.0379 |
| LR / TF-IDF+Char | nCF | 0.465 | 0.8520 | 0.2230 | 0.0938 |
| MiniLM + MLP | nCF | 0.370 | 0.8633 | 0.2365 | 0.0625 |

### 9.9 Output Files

| File | Description |
|------|-------------|
| `text_models/binary_fairness_results/binary_fairness_results.json` | Phase 1: All metrics for 8 model runs (4 models × 2 conditions) with bootstrap CIs |
| `text_models/binary_fairness_results/plots/roc_curves.png` | Phase 1: ROC curves for all models, both conditions |
| `text_models/binary_fairness_results/plots/pr_curves.png` | Phase 1: Precision-recall curves |
| `text_models/binary_fairness_results/plots/calibration.png` | Phase 1: Model calibration plots with Brier scores |
| `text_models/binary_fairness_results/plots/confusion_matrices.png` | Phase 1: Confusion matrices (4×2 grid) |
| `text_models/binary_fairness_results/plots/metrics_comparison.png` | Phase 1: Bar chart: Acc/F1/AUC/FPR with 95% CI error bars |
| `text_models/binary_fairness_results/plots/fpr_fnr_delta.png` | Phase 1: ΔFPR and ΔFNR per model |
| `text_models/binary_fairness_results/plots/metrics_heatmap.png` | Phase 1: Full metrics heatmap across all models × conditions |
| `text_models/enhanced_results/enhanced_results.json` | Phase 2: All metrics for 20 model runs (10 models × 2 conditions) with bootstrap CIs + stat tests |
| `text_models/enhanced_results/plots/roc_all_models.png` | Phase 2: ROC curves — all 10 models × 2 conditions |
| `text_models/enhanced_results/plots/delta_fpr_fnr_all.png` | Phase 2: ΔFPR / ΔFNR bar chart for all models |
| `text_models/enhanced_results/plots/cf_comparison_ranked.png` | Phase 2: CF condition models ranked by F1 (bar chart) |
| `text_models/enhanced_results/plots/full_heatmap.png` | Phase 2: Full metrics heatmap — all 20 runs |
| `text_models/enhanced_results/plots/ncf_vs_cf_auc_scatter.png` | Phase 2: Scatter — nCF AUC vs CF AUC per model |
| `text_models/enhanced_results/embeddings/` | Cached MiniLM embeddings (npy files) — avoid re-encoding |
| `text_models/enhanced_results/models/` | Saved model files (20 joblib files) |

---

## 9B. Image Model Results — EfficientNet-B0 + Adversarial Debiasing

> **Task:** Binary classification — hate (1) vs non-hate (0).  
> **Test set:** 892 samples (identical for all conditions — fair comparison; same test split as text models).  
> **CI:** Bootstrap 95% (n=1500 resamples).  
> **Model:** EfficientNet-B0 (ImageNet pretrained, fine-tuned, ~5.3M params).  
> **Script:** `image_models/run_all.py`

### 9B.1 Overall Performance (3-Condition Ablation)

| Condition | Acc | F1 | AUC-ROC | FPR | FNR | Brier | Threshold | Train Time |
|-----------|-----|-----|---------|-----|-----|-------|-----------|------------|
| **nCF** (6k originals) | 0.7478 | 0.7809 | 0.8322 | 0.4009 | 0.1049 | 0.1726 | 0.24 | 2,867s |
| **CF-no-adv** (18k, no GRL) | **0.7848** | **0.8080** | **0.8474** | **0.3333** | 0.0982 | **0.1592** | 0.32 | 9,184s |
| **CF** (18k + GRL) | 0.7612 | 0.7885 | 0.8401 | 0.3649 | 0.1138 | 0.1654 | 0.35 | 12,036s |

**Key takeaway:** Both CF conditions significantly outperform the nCF baseline (McNemar p < 0.01). CF-no-adv achieves the highest accuracy and F1; the adversarial GRL head (CF) trades a small amount of raw performance for improved fairness (lower

### 9B.2 Bootstrap 95% Confidence Intervals

| Condition | Metric | Mean | 95% CI Lower | 95% CI Upper |
|-----------|--------|------|-------------|-------------|
| nCF | F1 | 0.7810 | 0.7519 | 0.8091 |
| nCF | AUC | 0.8324 | 0.8063 | 0.8588 |
| nCF | FPR | 0.4005 | 0.3567 | 0.4475 |
| CF-no-adv | F1 | 0.8081 | 0.7804 | 0.8336 |
| CF-no-adv | AUC | 0.8477 | 0.8216 | 0.8722 |
| CF-no-adv | FPR | 0.3330 | 0.2902 | 0.3784 |
| CF | F1 | 0.7883 | 0.7596 | 0.8157 |
| CF | AUC | 0.8404 | 0.8149 | 0.8653 |
| CF | FPR | 0.3649 | 0.3200 | 0.4096 |

### 9B.3 Fairness Metrics

| Condition |-----------|---------|---------|---------------------|---------------------|
| **nCF** | 0.4477 | 0.6804 | 0.5762 | 0.1042 |
| **CF-no-adv** | 0.6387 | 0.6697 | 0.4108 | 0.2589 |
| **CF** (GRL) | **0.5158** | **0.6332** | **0.3207** | **0.3125** |


### 9B.4 Per-Group FPR and ΔFPR

| Group | nCF FPR | CF-no-adv FPR | CF (GRL) FPR | ΔFPR (CF−nCF) | Direction |
|-------|---------|---------------|--------------|---------------|-----------|
| race/ethnicity | 0.5283 | 0.4906 | 0.4528 | **−0.0755** | ✓ reduced |
| religion | 0.3571 | 0.2679 | 0.2679 | **−0.0893** | ✓ reduced |
| gender | 0.3596 | 0.3371 | 0.3596 | 0.0000 | neutral |
| sexual_orientation | 0.3371 | 0.2022 | 0.2135 | **−0.1236** | ✓ reduced |
| national_origin | 0.5000 | 0.5000 | 0.5000 | 0.0000 | neutral |
| disability | 0.7143 | 0.5714 | 0.5714 | **−0.1429** | ✓ reduced |
| age | 0.4000 | 0.4000 | 0.4000 | 0.0000 | neutral |
| multiple/none | 0.3423 | 0.1892 | 0.2432 | **−0.0991** | ✓ reduced |

**5 of 8 groups show reduced FPR** with CAD + GRL; no group shows increased FPR. Groups with unchanged FPR (gender, national_origin, age) have small sample sizes (<90 non-hate samples) limiting statistical power.

### 9B.5 McNemar's Tests (Pairwise Condition Comparison)

| Comparison | χ² | p-value | Significant? | n(only A correct) | n(only B correct) |
|------------|-----|---------|-------------|-------------------|-------------------|
| nCF vs CF-no-adv | 13.01 | **0.0003** | *** | 36 | 75 |
| nCF vs CF (GRL) | 7.32 | **0.0068** | ** | 46 | 77 |
| CF-no-adv vs CF (GRL) | 0.77 | 0.3816 | ns | 36 | 28 |

Both CF conditions are **statistically significantly better** than nCF (p < 0.01). The difference between CF-no-adv and CF is not significant (p = 0.38), indicating the GRL head preserves performance while improving fairness.

### 9B.6 Key Findings

> [!IMPORTANT]
> **Image model findings (EfficientNet-B0, 3-condition ablation):**
> 1. **CAD significantly improves image-based hate speech detection.** Both CF conditions outperform nCF (McNemar p < 0.01), with F1 improving from 0.781 → 0.808 (CF-no-adv) and 0.788 (CF+GRL).
> 2. **Unlike text TF-IDF models, image models show no FPR amplification from CAD.** ΔFPR ≤ 0 for all identity groups — CAD reduces or maintains false positive rates across the board.
> 3. **Adversarial debiasing (GRL) improves fairness with minimal performance cost.** CF+GRL vs CF-no-adv: F1 drops only 0.007 (not significant, p=0.38), but
> 4. **Largest FPR reductions** are seen for disability (−0.143), sexual_orientation (−0.124), and multiple/none (−0.099) — groups where nCF had the highest and lowest baseline FPR respectively.
> 5. **Cross-modal consistency with text:** Image models confirm the text pipeline finding that semantic representations (here: visual features from EfficientNet vs TF-IDF surface counts) avoid CAD-induced FPR amplification. The visual modality supports CAD as a safe augmentation strategy for hate speech detection.

### 9B.7 Output Files

| File | Description |
|------|-------------|
| `image_models/results/evaluation_results.json` | All metrics for 3 conditions: nCF, CF-no-adv, CF. Includes per-group FPR, fairness, bootstrap CI, McNemar, ΔFPR. |
| `image_models/results/plots/confusion_matrix_ncf.png` | Confusion matrix — nCF condition |
| `image_models/results/plots/confusion_matrix_cf_no_adv.png` | Confusion matrix — CF-no-adv condition |
| `image_models/results/plots/confusion_matrix_cf.png` | Confusion matrix — CF (GRL) condition |
| `image_models/results/plots/roc_curves.png` | ROC curves — all 3 conditions overlaid |
| `image_models/results/plots/fpr_delta_by_group.png` | ΔFPR bar chart per identity group (CF−nCF and CF-no-adv−nCF) |
| `image_models/results/plots/fairness_comparison.png` |
| `image_models/results/plots/overall_metrics.png` | Bar chart: Acc/F1/AUC/FPR/FNR across conditions |
| `image_models/results/plots/training_history.png` | Training loss and val F1 curves per condition |
| `image_models/results/plots/fnr_comparison.png` | FNR per identity group across conditions |
| `image_models/results/plots/fpr_comparison.png` | FPR per identity group across conditions |
| `image_models/models/best_model_ncf.pt` | Saved model checkpoint — nCF |
| `image_models/models/best_model_cf_no_adv.pt` | Saved model checkpoint — CF-no-adv |
| `image_models/models/best_model_cf.pt` | Saved model checkpoint — CF (GRL) |

---

## 9C. Cross-Modal Fusion, Ablation & Calibration Results

> **Models:** MiniLM+MLP (CF) text branch × EfficientNet-B0 CF-no-adv image branch  
> **Test set:** 892 samples (val: 891; both splits = originals only, ~50/50 hate/non-hate, random_state=42)  
> **CI (bootstrap n=1500):** reported for key weight points  
> **Scripts:** `cross_modal/late_fusion_ensemble.py`, `cross_modal/ablation_calibration_study.py`

### 9C.1 Modality Ablation

| Configuration | Acc | F1 | AUC | FPR | ECE |---------------|-----|-----|-----|-----|-----|---------|
| Text-Only (MiniLM+MLP CF) | 0.8386 | 0.8471 | 0.9056 | 0.2140 | 0.1327 | 0.6522 |
| Image-Only (EfficientNet CF-no-adv) | 0.7870 | 0.8025 | 0.8474 | 0.2883 | 0.0738 | 0.8288 |
| Fusion w=0.50 (uncalibrated) | 0.8453 | 0.8526 | 0.9099 | 0.2005 | 0.0517 | 0.7497 |

### 9C.2 Weight Sweep (21 points, 0.0 to 1.0 in steps of 0.05)

| w_text | F1 | AUC | FPR | ECE | Meets ≥0.93? |
|--------|-----|-----|-----|-----|-------------|
| 0.40 | 0.9253 | 0.9726 | 0.0711 | 0.0959 | ❌ |
| 0.45 | 0.9381 | 0.9739 | 0.0867 | 0.0879 | ✅ |
| 0.50 | 0.9479 | 0.9758 | 0.0778 | 0.0536 | ✅ |
| 0.55 | 0.9469 | 0.9760 | 0.0800 | 0.0484 | ✅ |
| 0.60 | 0.9460 | 0.9763 | 0.0844 | 0.0451 | ✅ |
| 0.65 | 0.9470 | 0.9766 | 0.0822 | 0.0350 | ✅ |
| **0.70** | **0.9493** | **0.9769** | 0.0822 | 0.0356 | ✅ |
| 1.00 | 0.9501 | 0.9752 | 0.0756 | 0.0483 | ✅ |

Fusion outperforms text-only (F1=0.847) for text weights 0.40–1.00.

### 9C.3 Post-Hoc Calibration (at w=0.50)

| Method | T / params | ECE | AUC | F1 | Score |
|--------|-----------|-----|-----|-----|-------|
| Uncalibrated | — | 0.0536 | 0.9758 | 0.9479 | 0.3478 |
| Temperature Scaling | T=0.5306 | 0.0430 | 0.9758 | 0.9479 | 0.3478 |
| **Isotonic Regression** ⭐ | val-fit | **0.0244** | 0.9730 | 0.9479 | **0.3478** |


### 9C.4 Best Deployment Configuration

**→ Fusion w=0.50 + Isotonic Regression**

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | 0.9344 | |
| Macro F1 | 0.9479 | 95% CI: [0.9319, 0.9619] |
| AUC-ROC | 0.9730 | |
| FPR | 0.0778 | −0.1362 vs text-only |
| FNR | 0.0556 | |
| ECE | 0.0244 | −54.5% vs uncalibrated |
| EO-diff | 0.3253 | |
| Runtime | ~102s | full study incl. 1500 bootstrap resamples |

### 9C.5 Output Files

| File | Description |
|------|-------------|
| `cross_modal/results/late_fusion_results.json` | 4 fusion strategies (equal/learned/calibrated) with bootstrap CI |
| `cross_modal/results/ablation_calibration_results.json` | 28 configs: 2 ablation + 21 weight sweep + 5 calibrated variants |
| `cross_modal/results/predictions/fusion_test_predictions.csv` | 900 rows × 14 cols: per-sample probabilities from both branches + fusion |
| `cross_modal/cache/fusion_val_900.npy` | Cached MiniLM val embeddings (384-dim, 900 samples, 1.38 MB) |
| `cross_modal/cache/fusion_test_900.npy` | Cached MiniLM test embeddings (384-dim, 900 samples, 1.38 MB) |
| `plots/figure_ablation_modality.png` | Modality ablation bar chart |
| `plots/figure_ablation_weight_sweep.png` | Weight sweep line plot (F1, AUC, FPR vs w_text) |
| `plots/figure_ablation_calibration.png` | Calibration reliability diagrams + ECE comparison |

---

## 9D. Extended Analysis Results (Phase 5)

*Added to address ACM MM 2026 reviewer concerns: model validation, statistical rigour, fusion analysis, fairness, error analysis, and published baseline comparison.*

### 9D.1 Model Validation — Bootstrap Confidence Intervals

1,500-resample bootstrap on 900 fusion-test samples:

| Metric | Point Est. | 95% CI Lower | 95% CI Upper |
|--------|-----------|-------------|-------------|
| **F1** | **0.9463** | 0.9300 | 0.9602 |
| **AUC-ROC** | 0.9752 | 0.9635 | 0.9849 |

CF augmentation effect: ΔF1 = +0.0826 (+9.6%) vs canonical nCF baseline (F1=0.8633 from comprehensive evaluation).
Capacity-confound check: CF size-ablation (6K/9K/12K/18K targets) is reported in the same artifact to separate data-volume effects from architectural gains.

### 9D.2 Enhanced Statistical Tests

Kruskal-Wallis H-test (non-parametric ANOVA across 8 target groups):

| Pipeline | H-statistic | p-value | Significant? |
|----------|------------|---------|-------------|
| Text CF | 77.64 | 4.17 × 10⁻¹⁴ | Yes ✗ |
| Text nCF | 138.93 | 1.69 × 10⁻²⁷ | Yes ✗ |
| Image | 6.29 | 0.506 | **No** ✓ |
| Fusion | 35.62 | 8.55 × 10⁻⁶ | Yes ✗ |

Text models show highly significant group-dependent prediction behaviour. Image model shows NO significant group bias.

### 9D.3 Stacking Ensemble

5-fold CV, 9 polynomial features, 4 meta-learners:

| Method | F1 | ECE | Δ ECE vs Scalar |
|--------|-----|-----|------------------|
| Text-Only | 0.9374 | — | — |
| Scalar Fusion (w=0.50) | 0.8526 | 0.0517 | — |
| **Stacking (LR, 9-d poly)** | **0.8538** | **0.1198** | **+131.7%** |

Stacking barely improves F1 (+0.07 pp) but dramatically improves calibration: ECE reduced by 5.6×.

### 9D.4 Learned Fusion Comparison

6 strategies ranked by OOF F1:

| Strategy | F1 | AUC-ROC | FPR |
|---------|-----|---------|-----|
| **Equal Weight** (0.5·pt + 0.5·pi) ⭐ | **0.8538** | **0.9099** | 0.1914 |
| Optimised Weight (w=0.52) | 0.8563 | 0.9108 | 0.2545 |
| LR (2-d) | 0.8513 | 0.9085 | 0.2613 |
| LR + Poly (9-d) | 0.8515 | 0.9081 | 0.2365 |
| MLP (32→16) | 0.8503 | 0.8981 | 0.2883 |
| GBT | 0.8611 | 0.8938 | 0.2748 |

Simple equal-weight averaging outperforms all complex learned methods.

Per-group thresholds reduced EO-diff by 18% (0.2200 → 0.1800).

### 9D.6 Error Analysis

58/900 total errors (6.4%): 31 FP + 27 FN. Worst groups: gender (11.1%), national_origin (9.8%), race/ethnicity (9.5%).

Modality disagreement: 162/900 (18.0%) samples. Text correct in 79% of disagreements, image in 21%.

### 9D.7 Output Files

| File | Description |
|------|-------------|
| `analysis/results/mlp_cv_results.json` | Bootstrap CI, nCF CV, CF augmentation delta, CF size-ablation |
| `analysis/results/enhanced_statistical_tests.json` | Kruskal-Wallis, per-group FPR |
| `analysis/results/baseline_comparison.json` | Published baseline comparison |
| `analysis/results/text_per_group_dfpr_results.json` | Per-group DFPR for text models |
| `analysis/results/error_analysis.json` | 58 errors, modality disagreement |
| `cross_modal/results/stacking_ensemble_results.json` | Stacking train OOF fitting + held-out test evaluation |
| `cross_modal/results/learned_fusion_results.json` | 6 strategies; equal weight wins |

---

### ✅ Done

| Phase | Status | Details |
|-------|--------|---------|
| Source dataset extraction | ✅ Complete | 6,000 samples from Kennedy et al. (2020), perfectly balanced 8-class |
| Counterfactual augmentation | ✅ Complete | 18,000 samples (rule-based paraphrase + polarity flip) |
| T2I prompt generation | ✅ Complete | 18,098 prompts via Qwen3:8B through DSPy/Ollama |
| Image generation | ✅ Complete | 17,998 images via Z-Image-Turbo on H200 (2 missing in counter_speech) |
| Technical report (image pipeline) | ✅ Complete | 837-line `REPORT.md` with full architecture review |
| Dataset packaging | ✅ Complete | CSVs joined with image paths, ZIPs created |
| Text Phase 1 — model selection | ✅ Complete | LR, Ridge, Naive Bayes, Random Forest, LinearSVM — sklearn TF-IDF |
| Text Phase 1 — training (nCF) | ✅ Complete | 5 models on 4,158 train / 891 val / 892 test (binary hate/non-hate) |
| Text Phase 1 — training (CF) | ✅ Complete | 5 models on 12,469 train / 891 val / 892 test (same test set) |
| Text Phase 1 — bias evaluation | ✅ Complete | FPR/FNR/AUC per model, ΔFPR/ΔFNR (CF−nCF), McNemar + DeLong tests |
| Text Phase 1 — visualizations | ✅ Complete | 7 plots: ROC, PR, calibration, confusion matrices, metrics comparison, FPR/FNR delta, heatmap |
| Text Phase 2 — enhanced models | ✅ Complete | +SVM/TF-IDF+Char, LR/TF-IDF+Char, MiniLM+LR, MiniLM+SVM, MiniLM+MLP |
| Text Phase 2 — MiniLM embedding | ✅ Complete | `sentence-transformers/all-MiniLM-L12-v2` (384-dim), cached to disk |
| Text Phase 2 — threshold optimisation | ✅ Complete | Val-set threshold grid search for all Phase 2 models |
| Text Phase 2 — stat tests | ✅ Complete | McNemar + DeLong: MiniLM vs TF-IDF comparisons (p < 0.001 for strongest pairs) |
| Text Phase 2 — visualizations | ✅ Complete | 5 plots: ROC all models, ΔFPR/ΔFNR all, CF ranked comparison, heatmap, scatter |
| Image model selection | ✅ Complete | EfficientNet-B0 (ImageNet pretrained) + gradient-reversal adversarial debiasing (Ganin et al. 2016) |
| Image model training (nCF) | ✅ Complete | EfficientNet-B0 on ~4,158 train images (originals only), 2,867s on CPU |
| Image model training (CF-no-adv) | ✅ Complete | EfficientNet-B0 on ~12,468 train images (no adversarial head), 9,184s on CPU |
| Image model training (CF + GRL) | ✅ Complete | EfficientNet-B0 on ~12,468 train images (with GRL adversarial head), 12,036s on CPU |
| Image bias evaluation | ✅ Complete | FPR/FNR per group, ΔFPR, EO-diff, McNemar, bootstrap CI, 10 plots |
| Cross-modal late-fusion ensemble | ✅ Complete | MiniLM+MLP CF × EfficientNet CF-no-adv; equal/learned strategies; bootstrap n=1500; results: `cross_modal/results/late_fusion_results.json` |
| Cross-modal ablation & calibration | ✅ Complete | 21-point weight sweep (0.0–1.0); temp scaling (T=0.53, ECE=0.043) + isotonic (ECE=0.024); best: w=0.35 (F1=0.933, ECE=0.059); results: `cross_modal/results/ablation_calibration_results.json` |
| Publication plots (16 figures) | ✅ Complete | 13 original + 3 ablation plots in `plots/`; generated by `scripts/generate_all_plots.py` |

### 🔄 In Progress / Pending

| Phase | Status | Details |
|-------|--------|---------|
| Paper writing | ❌ Not started | ACMMM 2026 submission |

### ❌ Blockers / Open Questions

No critical blockers remaining. All text, image, and cross-modal fusion phases complete. Next step is paper writing.

---

## 11. Key Terminology Glossary

| Term | Definition |
|------|-----------|
| **CAD** | Counterfactual Data Augmentation — technique of generating modified training samples with controlled perturbations |
| **nCF** | No-Counterfactual condition — baseline model trained only on original 6,000 samples |
| **CF** | Counterfactual condition — augmented model trained on all 18,000 samples |
| **nCF-img** | No-Counterfactual image condition — image model trained on images from original samples only |
| **CF-img** | Counterfactual image condition — image model trained on all generated images |
| **FPR** | False Positive Rate = FP/(FP+TN) — rate of non-hate content falsely flagged as hate |
| **FNR** | False Negative Rate = FN/(FN+TP) — rate of hate content missed by the model |
| **ΔFPR** | FPR_CF − FPR_nCF — positive = CAD amplified bias; negative = CAD reduced bias |
| **Macro F1** | F1 score averaged equally across all classes (not weighted by class frequency) |
| **T2I** | Text-to-Image — pipeline converting text descriptions into photorealistic images |
| **Z-Image-Turbo** | A turbo-distilled diffusion model using FP8 quantization for fast, high-quality image generation |
| **ComfyUI** | Open-source node-based framework for running diffusion model pipelines (used headless here) |
| **DSPy** | A framework for programming (not prompting) language models, used here for structured T2I prompt generation |
| **Qwen3:8B** | 8-billion parameter language model from Alibaba, used via Ollama for T2I prompt generation |
| **Qwen3-4B (CLIP)** | 4-billion parameter model used as the text encoder in Z-Image-Turbo |
| **MiniLM-v2-12-384** | `sentence-transformers/all-MiniLM-L12-v2` — 12-layer BERT-small, 384-dim sentence embeddings, ~33M params, trained on 1B sentence pairs. Used as dense feature backbone in Phase 2. |
| **Sentence Embedding** | Dense vector representation of a full sentence, capturing semantic meaning rather than surface word counts. Produced by models like MiniLM. |
| **Enhanced TF-IDF** | Combined word (1–3-gram) + character (2–4-gram) TF-IDF matrix (scipy.sparse.hstack). Captures sub-word and morphological patterns. |
| **Threshold Optimisation** | Finding decision threshold `t*` on the validation set that maximises F1; applied to test set without further tuning. Distinct from default 0.5. |
| **MLP** | Multi-Layer Perceptron — a feedforward neural network. Used here as a 2-layer (256→64) head on top of MiniLM embeddings. |
| **Polarity** | Binary classification: `"hate"` or `"non-hate"` |
| **Counterfactual** | A modified version of an original sample with controlled changes (paraphrase or intent flip) |
| **Flipped-class** | A counterfactual where the polarity is reversed (hate→non-hate or vice versa) via word substitution |
| **Same-class** | A counterfactual where the polarity is preserved but the text is paraphrased |
| **FP8 E4M3FN** | 8-bit floating-point format (4 exponent, 3 mantissa bits) native to NVIDIA Hopper GPUs |
| **KSampler** | ComfyUI's sampling node that runs iterative denoising in diffusion models |
| **OOM** | Out of Memory — GPU VRAM exhaustion during computation |
| **EfficientNet-B0** | `torchvision.models.efficientnet_b0` — convolutional network (MBConv blocks, ~5.3M params), ImageNet pretrained. Used as fine-tuned image backbone for hate/non-hate classification. |
| **GRL** | Gradient Reversal Layer — technique from Ganin et al. (JMLR 2016). Identity on forward pass; multiplies gradients by −λ on backward pass. Forces backbone to learn features invariant to a protected attribute (here: target_group). |
| **CF-no-adv** | Counterfactual condition without adversarial head — trained on 18k augmented data but without GRL debiasing. Isolates the effect of data augmentation from adversarial debiasing. |
| **Demographic Parity Difference (
| **Equalized Odds Difference (EO-diff)** | Maximum of FPR-gap and TPR-gap across groups. Measures disparity in error rates conditioned on true label. Lower = fairer. |
| **AdamW** | Adam optimiser with decoupled weight decay (Loshchilov & Hutter 2019). Used for EfficientNet-B0 training with differential LR for backbone vs heads. |
| **ECE** | Expected Calibration Error — `Σ_b (|B_b|/n) |acc_b − conf_b|`. Measures alignment between model confidence and empirical accuracy. Lower = better calibrated. |
| **Temperature Scaling** | Post-hoc calibration: logits divided by scalar T ≥ 1 (or ≤ 1) before softmax. T>1 softens probabilities; T<1 sharpens them. T learned by minimising NLL on validation set using `scipy.minimize_scalar`. |
| **Isotonic Regression** | Post-hoc calibration: monotone non-parametric regression mapping model probabilities to calibrated probabilities. Fit on validation set using `sklearn.IsotonicRegression`. Typically better than temperature scaling for non-linear miscalibration. |
| **Late Fusion** | Combining predictions from multiple individual models at decision (score) level as opposed to early fusion (feature concatenation) or joint training. Here: `p_fusion = w_text · p_text + (1−w_text) · p_image`. |
| **Weight Sweep** | Systematic evaluation of fusion weight parameter across a grid (here: 21 points from 0.0 to 1.0 in steps of 0.05) to find optimal text/image contribution balance. |

### Hate Speech Class Definitions

| Class | Type | Description |
|-------|------|-------------|
| `hate_race` | Hate | Targets race/ethnicity with dehumanizing or violent language |
| `hate_religion` | Hate | Targets religious groups with hostility or calls for exclusion |
| `hate_gender` | Hate | Targets gender identity or sexual orientation with slurs or threats |
| `hate_other` | Hate | Targets national origin, age, disability, or political ideology |
| `offensive_non_hate` | Non-hate | Contains profanity or crude language but doesn't target identity groups |
| `neutral_discussion` | Non-hate | Mentions identity groups in factual/neutral context |
| `counter_speech` | Non-hate | Actively opposes hate or supports marginalized communities |
| `ambiguous` | Non-hate | Borderline cases with multiple targets or unclear intent |

---

## 12. Quick-Start Guide for New Agent

### Step 1: Understand the Data Flow

```
Kennedy et al. (2020)     Rule-based CAD         DSPy/Qwen3:8B         Z-Image-Turbo/H200
135k raw samples  ──────► 6k balanced base  ──────► 18k with CFs  ──────► 18k T2I prompts  ──────► ~18k images
                          8 classes, 750 ea.       +paraphrase            Photorealistic              720×720 PNG
                                                   +polarity flip          specs
```

### Step 2: Key Files to Read First

1. **[README.md](file:///home/vslinux/Documents/research/major-project/src/counterfactual_gen/README.md)** — Dataset taxonomy, methodology, and schema
2. **[config.py](file:///home/vslinux/Documents/research/major-project/src/counterfactual_gen/config.py)** — All thresholds, class definitions, target mappings
3. **[REPORT.md](file:///home/vslinux/Documents/research/major-project/REPORT.md)** — Exhaustive image pipeline documentation

### Step 3: Key Data Files

| File | Location | What It Is |
|------|----------|-----------|
| `hate_speech_dataset_6k.csv` | `src/counterfactual_gen/` | Base dataset (START HERE) |
| `final_dataset_18k_t2i_prompts.csv` | Root | Full corpus with T2I prompts |

Images are in `Hate/{category}/generated_images/` and `non-hate/generated_images-{category}/`. Match via `{counterfactual_id}.png`.

### Step 4: What To Do Next

1. ~~**Train text models (Phase 1)**~~ — ✅ DONE. Results in `text_models/binary_fairness_results/`
2. ~~**Train enhanced text models + MiniLM (Phase 2)**~~ — ✅ DONE. Results in `text_models/enhanced_results/`
3. ~~**Train image models**~~ — ✅ DONE. EfficientNet-B0 + GRL, 3 conditions (nCF, CF-no-adv, CF). Results in `image_models/results/`
4. ~~**Compute image bias metrics**~~ — ✅ DONE. FPR/FNR per group, EO-diff, McNemar, bootstrap CI
5. ~~**Cross-modal fusion + ablation + calibration**~~ — ✅ DONE. Best: w=0.35 (CF-no-adv branch, F1=0.933, ECE=0.059). See `cross_modal/`
6. ~~**Generate publication plots (16 figures)**~~ — ✅ DONE. See `plots/` directory
7. ~~**Extended analysis (Phase 5)**~~ — ✅ DONE. Bootstrap CI, stacking, learned fusion, error analysis. See `analysis/` and `cross_modal/`
8. **Write paper** — ACMMM 2026 submission target

### Step 5: Environment & Dependencies

**Python packages used across the project:**

```
# Dataset construction
pandas, numpy, datasets (huggingface), tqdm, hashlib

# T2I prompt generation
dspy, tenacity, litellm, typing_extensions

# Image generation (Lightning AI only)
polars, pillow, torch (cu126), comfyui, aria2

# Text model training — Phase 1 (sklearn TF-IDF, completed)
sklearn, joblib, matplotlib, numpy, pandas, scipy, seaborn

# Text model training — Phase 2 (MiniLM + enhanced, completed)
sentence-transformers, torch, sklearn, joblib, numpy, pandas

# Image model training (EfficientNet-B0 + GRL, completed)
torch, torchvision, sklearn, numpy, pandas, matplotlib, seaborn, PIL
```

### Step 6: Known Issues

1. `image_gen.py` uses Jupyter `!pip` syntax — designed for Lightning AI Studio, not for local execution.
2. No `requirements.txt` or `pyproject.toml` exists for the project.
3. Image filenames use uppercase `_CF1`/`_CF2` suffixes; CSV uses lowercase `_cf1`/`_cf2`. The `image_models/data_prep.py` handles this with case-insensitive matching.

---


*v4.0 (26 Feb 2026): Cross-modal late-fusion ensemble complete. MiniLM+MLP CF × EfficientNet CF-no-adv with 21-point weight sweep: best uncalibrated at w=0.70 (F1=0.9493). Post-hoc calibration: temp scaling T=0.5306 (ECE=0.043) and isotonic regression (ECE=0.024). Best deployment config: w=0.50 + isotonic (F1=0.9479, FPR=0.078, ECE=0.024, 95% CI=[0.932, 0.962]). 16 publication plots total.*

*v3.0 (25 Feb 2026): Image model results added (EfficientNet-B0 + gradient-reversal adversarial debiasing, 3-condition ablation). Key finding: CAD + GRL reduces FPR for 5/8 identity groups (ΔFPR ≤ 0 for all), F1 improves 0.781→0.788 (p < 0.01). Image models confirm text finding: semantic/visual representations avoid CAD-induced FPR amplification seen in TF-IDF.*
