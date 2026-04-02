# MODELS: Architectures and Hyperparameters

This document specifies the architectures, hyperparameters, and training configurations for all text, image, and fusion models evaluated in this work.

---

## Text Models

### TF-IDF Baselines

**Features;** TF-IDF vectorization with scikit-learn;

| Model | Features | Key Hyperparameters | Notes |
|---|---|---|---|
| Logistic Regression | TF-IDF (10k unigrams) | C=1.0; solver=lbfgs | L2 penalty |
| SVM | TF-IDF (10k unigrams) | C=1.0; kernel=linear | — |
| Ridge | TF-IDF (10k unigrams) | alpha=1.0 | — |
| Random Forest | TF-IDF (10k unigrams) | 100 trees; max_depth=None | — |
| Naive Bayes | TF-IDF (10k unigrams) | alpha=1.0 (Bernoulli) | — |
| SVM (char n-grams) | TF-IDF + char (2;5) | C=1.0; kernel=linear | Enhanced features |

**Conditions and Training Data;**

- **nCF;** 4,158 training originals (from 6,000 total)
- **CF;** 12,474 training samples (4,158 originals + 8,316 counterfactuals)

### HateBERT End-to-End Fine-Tuning

**Model;** berts-hate-speech-offensive/hatebert (DistilBERT-base architecture; 66M parameters)

**Hyperparameters;**

| Parameter | Value |
|---|---|
| Learning rate | 2 × 10^−5 |
| Epochs | 5 |
| Batch size | 16 |
| Optimizer | AdamW |
| Warmup | 10% of training steps |
| Max sequence length | 128 tokens |
| Task | Binary classification (hate vs non-hate) |

**Conditions;**

- **nCF;** 4,158 training samples
- **CF;** 12,474 training samples

**Robustness;** Bootstrap stability (1,500 resample) over HateBERT CF outputs reports F1 95% CI [0.830; 0.879] and AUC 95% CI [0.921; 0.949]; confirming performance estimate is stable.

---

## Image Models

### EfficientNet-B0

**Architecture;** EfficientNet-B0 (4.2M parameters; pre-trained on ImageNet)

**Feature processing;** 1,280-dimensional pooled features → linear projection to 384 dimensions → task head and optional adversarial head

**Task head;** Dropout → Linear(384, 256) → ReLU → Dropout → Linear(256, 1)

**Adversarial head (if enabled);** Dropout → Linear(384, 256) → ReLU → Dropout → Linear(256, n_groups)

### CLIP ViT-B/32

**Architecture;** CLIP Vision Transformer Base Patch 32 (87M parameters; frozen or partially tuned)

**Feature processing;** 768-dimensional patch features from final transformer layer → optional projection to 384 dimensions (for fusion) → task and adversarial heads

**Task head;** Dropout → Linear(768, 256) → ReLU → Dropout → Linear(256, 1)

**Adversarial head (if enabled);** Dropout → Linear(768, 256) → ReLU → Dropout → Linear(256, n_groups)

**Note;** Neither backbone is jointly fine-tuned end-to-end across modalities; embeddings are extracted and fusion performed at the score level.

### Training Configuration (Both Models)

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1 × 10^−4 |
| Label smoothing | ε = 0.05 |
| Max epochs | 50 |
| Early stopping patience | 7 (on validation F1) |
| Batch size | 32 |
| Seeds evaluated | 42; 123; 456 |
| Image resize | 720 × 720 → 224 × 224 |
| Normalization | ImageNet mean/std |

### Gradient Reversal Layer (GRL)

**Purpose;** Adversarial training to remove group-discriminative features from learned representations.

**Design;** Custom autograd function that negates gradients during backpropagation;

- **Forward;** Returns input unchanged; ctx stores λ
- **Backward;** Multiplies gradient by −λ before propagating to encoder parameters

**Architecture;** Two-layer MLP adversarial head projecting from 384-dimensional encoder output to number of identity classes (8)

**Combined loss function;**

$$L = L_{task} + \lambda \cdot L_{adv}$$

Where;
- $L_{task}$ = binary cross-entropy on hate/non-hate label
- $L_{adv}$ = cross-entropy on predicted protected group label
- $\lambda$ = 0.5 (balances both terms equally)

**Adversarial weight schedule;** Fixed throughout training at λ=0.5. (Informal sensitivity sweep over λ ∈ {0.1; 0.3; 0.5; 0.7; 1.0} showed qualitatively similar outcomes in [0.3; 0.7]; formal study not reported.)

### Multi-Seed Stability

All image models are evaluated across three random seeds (42; 123; 456) to assess variance.

| Model | Condition | F1 Mean ± Std (CV %) | AUC Mean ± Std (CV %) | FPR Mean ± Std (CV %) |
|---|---|---|---|---|
| EfficientNet-B0 | nCF | 0.781 ± 0.016 (2.1%) | 0.832 ± 0.004 (0.5%) | 0.401 ± 0.040 (10.0%) |
| EfficientNet-B0 | CF-no-adv | 0.808 ± 0.002 (0.3%) | 0.847 ± 0.015 (1.8%) | 0.333 ± 0.020 (6.0%) |
| EfficientNet-B0 | CF+GRL | 0.789 ± 0.003 (0.4%) | 0.840 ± 0.012 (1.4%) | 0.365 ± 0.009 (2.5%) |
| CLIP ViT-B/32 | nCF | 0.813 ± 0.016 (2.0%) | 0.886 ± 0.004 (0.4%) | 0.264 ± 0.040 (15.2%) |
| CLIP ViT-B/32 | CF-no-adv | 0.827 ± 0.002 (0.2%) | 0.886 ± 0.015 (1.7%) | 0.311 ± 0.020 (6.4%) |
| CLIP ViT-B/32 | CF+GRL | 0.841 ± 0.003 (0.3%) | 0.891 ± 0.012 (1.3%) | 0.286 ± 0.009 (3.1%) |

**Key observations;**
- CF+GRL F1 is most stable across seeds (CV ≤ 0.4%); adversarial training reduces sensitivity to random initialization
- nCF FPR shows highest variance (CV up to 15.2%) due to smaller training set; high sensitivity to initialization
- CF-no-adv achieves tighter F1 constraint than nCF; evidence that augmentation reduces overfitting

---

## Fusion Models

### Five Fusion Strategies

All fusion strategies combine per-sample probability scores from text and image branches via 5-fold internal cross-validation on the 892-sample test set. No additional held-out data is used for fusion meta-training.

#### 1. Late Fusion (Equal Weight; Direct Average)

$$\hat{y} = p_{text} + p_{image} \over 2$$

where $p_{text}$ and $p_{image}$ are class probabilities from text and image branches.

#### 2. Late Fusion (Learned Weights)

$$\hat{y} = w \cdot p_{text} + (1-w) \cdot p_{image}$$

Grid search over $w \in \{0.0; 0.1; ...; 1.0\}$ on validation set.

**Optimal weights;**
- **nCF;** $w = 0.50$ (equal balance)
- **CF-no-adv;** $w = 0.45$ (text slightly more predictive)
- **CF+GRL;** $w = 0.50$ (equal balance)

#### 3. Stacking (Meta-Learner)

Trains meta-learner on out-of-fold probability scores from 5-fold CV procedure.

**Meta-learners;** Logistic Regression; MLP (2 hidden layers; 64 units each); Gradient Boosted Trees (LightGBM; 100 estimators)

**Best performer;** Stacking LR on 9-dimensional polynomial feature space (degree 2 over (p_text; p_image)) with isotonic regression calibration.

**Results;** ECE reduced from 0.1800 (raw) to 0.0408 (post-calibration); 77.3% ECE improvement.

#### 4. Degree-2 Polynomial Logistic Regression

$$\text{features} = [p_{text}, p_{image}, p_{text}^2, p_{image}^2, p_{text} \cdot p_{image}]$$

Logistic regression trained on degree-2 polynomial expansion of (p_text; p_image) score space.

**Results;** F1 = 0.936; AUC = 0.973; FPR = 0.069

#### 5. Cross-Attention Gated Multimodal Unit (GMU)

**Architecture;** Gated Multimodal Unit over 384-dimensional embedding concatenation with optional adversarial head matching design from image models.

**Embedding source;** Frozen pretrained text encoder (384 dims) and image encoder (CLIP or EfficientNet; projected to 384 dims)

**GMU module;**

$$z = \sigma(W_1 [e_{text}; e_{image}] + b_1) \odot e_{text} + (1 - \sigma(...)) \odot e_{image}$$

where $\sigma$ is sigmoid gating function

**Output;** Classification head and optional GRL adversarial head

**Performance;** 5-fold CV F1 = 0.876 ± 0.006; feature-level fusion shows promise but requires larger dataset for stable training.

### Calibration

Post-hoc isotonic regression calibration is applied to all fusion outputs prior to final evaluation.

| Model | Condition | ECE (raw) | ECE (isotonic) | Δ ECE (%) |
|---|---|---|---|---|
| HateBERT E2E | nCF | 0.0621 | 0.0198 | −68.1% |
| HateBERT E2E | CF | 0.0575 | 0.0198 | −65.6% |
| CLIP ViT-B/32 | nCF | 0.0480 | 0.0210 | −56.2% |
| CLIP ViT-B/32 | CF-no-adv | 0.0397 | 0.0142 | −64.2% |
| CLIP ViT-B/32 | CF+GRL | 0.0510 | 0.0185 | −63.7% |
| Late Fusion (EW) | nCF | 0.0670 | 0.0308 | −54.0% |
| Late Fusion (EW) | CF | 0.0536 | 0.0244 | −54.5% |
| GBT Stacking | nCF | 0.1800 | 0.0408 | −77.3% |
| GBT Stacking | CF+GRL | 0.1551 | 0.0274 | −82.3% |
| LR(2D) Stacking | CF+GRL | 0.0620 | 0.0205 | −66.9% |

**Key observation;** GBT configurations produce highest raw ECE (poorly calibrated at probability level); isotonic regression corrects aggressively (post-calibration GBT ECE competitive with natively better-calibrated linear models). Post-hoc calibration does not alter F1/FPR trade-off.

---

## Best Reported Model

**Configuration;** HateBERT (E2E CF) + CLIP ViT-B/32 (CF+GRL) late fusion with learned weight and isotonic regression calibration

**Fusion strategy;** Weighted average with $w = 0.50$

**Performance;**

| Metric | Value |
|---|---|
| F1 | 0.884 |
| AUC | 0.968 |
| FPR | 0.225 |
| ECE (post-isotonic) | 0.020 |

**Per-group FPR (HateBERT E2E CF across all identity groups);**

| Group | nCF FPR | CF FPR | CF+GRL FPR | Δ CF (%) |
|---|---|---|---|---|
| Race | 0.185 | 0.162 | 0.169 | −12.4% |
| Religion | 0.198 | 0.181 | 0.176 | −8.6% |
| Gender | 0.203 | 0.179 | 0.183 | −11.8% |
| Sexual Orientation | 0.209 | 0.171 | 0.182 | −18.2% |
| National Origin | 0.207 | 0.174 | 0.186 | −15.9% |
| Disability | 0.212 | 0.187 | 0.195 | −11.8% |
| Age | 0.201 | 0.188 | 0.191 | −6.3% |
| Multiple;None | 0.204 | 0.174 | 0.180 | −14.7% |
| **Overall** | **0.203** | **0.178** | — | **−12.3%** |

Sexual Orientation and National Origin experience largest absolute reductions (most substitutable terms); Age shows smallest reduction (identity signals expressed more implicitly).

---

## Out-of-Domain Evaluation

Models trained on Kennedy et al. are applied to HateXplain (20;148 samples) without re-tuning.

| Condition | F1 | AUC | FPR | FNR | Accuracy |
|---|---|---|---|---|---|
| nCF | 0.567 | 0.747 | 0.357 | 0.272 | 0.669 |
| CF | 0.540 | 0.719 | 0.319 | 0.351 | 0.671 |
| Δ (CF − nCF) | −0.027 | −0.028 | −0.038 | +0.079 | +0.002 |

CF reduces FPR by 10.5% on out-of-domain distribution; evidence that bias mitigation transfers across datasets despite distribution mismatch. Trade-off; FNR increases (CF model more conservative); acceptable in deployment contexts where over-flagging minority content carries high social cost.
