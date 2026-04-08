## Supplementary File (Appendix)

See [Appendix.pdf](./Appendix.pdf) for extended results, dataset construction details, qualitative analysis, and statistical tests.

---

# COUNTER-HATE: Multimodal Counterfactual Augmentation for Bias-Aware Hate Detection

**Target Venue:** ACM Multimedia (Dataset Track) 2026  
**Status:** Reproducible; all pipelines documented

## MIDAS Dataset

**MIDAS** is a purpose-built dataset for evaluating bias mitigation in hate detection.

**Dataset Access:** [Available on Hugging Face](https://huggingface.co/datasets/vs16/counter-hate-dataset)
**Images:** [Access on Hugging Face](https://huggingface.co/datasets/vs16/counter-hate-dataset/tree/main/images)

### Composition

- **18,000 image-text pairs** across three configurable conditions;
  - **Originals (nCF):** 6,000 original text samples from Kennedy et al. (UC Berkeley D-Lab hate content collection)
  - **Counterfactuals (CF):** 12,000 rewritten text samples (2 per original) via Qwen2.5-7B-Instruct with deterministic fallback
  - **All 18,000 images:** Synthetically generated from text samples via Z-Image-Turbo (720×720 PNG, ~7.7 prompts/sec on H200), where 6,000 images generated from Original text samples; 12,000 images generated from Counterfactual Text pairs. 

### Identity Categories (8 total)

Eight protected group categories are represented; each original example is assigned one group label;

| Category | Representative Terms | Example Swap |
|---|---|---|
| Race / Ethnicity | Black, White, Asian, Hispanic | Black ↔ Asian |
| Religion | Muslim, Christian, Jewish, Hindu | Muslim ↔ Christian |
| Gender | women, men, girls, boys | women ↔ men |
| Sexual Orientation | gay, straight, LGBTQ+, lesbian | gay ↔ straight |
| National Origin | immigrant, refugee, Mexican | immigrant ↔ citizen |
| Disability | disabled, autistic, blind | disabled ↔ able-bodied |
| Age | elderly, young, teen | elderly ↔ young |
| Multiple / None | compound identity or absent | context-driven rewrite |

### Stratification and Splits

Canonical stratified splits at the original_sample_id level (seed=42);

| Partition | Original IDs | Total Rows (CF inc.) | Ratio |
|---|---|---|---|
| Train | 4,158 | 12,474 | 69.99% |
| Val | 891 | 891 | 15.00% |
| Test | 892 | 892 | 15.01% |
| **Total** | **5,941** | **14,257** | **100%** |

Note: Validation and test sets contain original samples only; counterfactual variants are confined to the training partition of their source original to prevent leakage.

### Counterfactual Generation Pipeline

Text rewriting proceeds via Qwen2.5-7B-Instruct (Kaggle T4×2 GPU):

1. **Detect identity terms** via regex dictionaries (race, religion, gender, sexuality, national origin, disability, age)
2. **Two prompt modes**:
   - **Explicit mode** (identity terms detected); Build swap prompt to substitute target group term from same category
   - **Implicit mode** (contextual identity); Build rewrite prompt for full paraphrase with shifted demographic reading
3. **Generation params**: temperature=0.25, top-p=0.9, max_new_tokens=128, repetition_penalty=1.1
4. **Post-processing**:
   - Strip &lt;think&gt; tokens and LLM preamble artifacts
   - Apply CJK character guard; discard outputs with non-ASCII characters outside expected set
   - Validate; check length ratio, label-preserving structure, absence of polarity-correlated descriptors
   - Fallback; if validation fails, apply deterministic regex-based term substitution
5. **Result**: Corpus of 18,000 samples; 6,000 originals + 12,000 counterfactuals with exactly three rows per original_sample_id

### Image Synthesis

All 18,000 images are synthetically generated from text samples (not collected from social media). Each text sample is passed to Z-Image-Turbo as a structured visual prompt;

- **Model**: Z-Image-Turbo (FP8 diffusion; Qwen3-4B CLIP backbone)
- **Parameters**: 720×720 resolution, 9 Euler diffusion steps, CFG scale 1.0
- **Throughput**: ~7.7 prompts/sec on Lightning AI H200 (141 GB VRAM); 39 hours for full corpus
- **Prompt enhancement**: Fixed quality prefix + row-level scene description + strict anti-text negative block
- **Deterministic seeding**: seed = (0xDEADBEEF + i × 1,000,003) mod 2^32 for reproducibility
- **Quality control**: Visual tone bias audit on 200-sample subset; confirmed no statistically significant difference in mean brightness or saturation between hate and non-hate classes

For details on identity mapping, dataset construction challenges, and counterfactual examples per class, See [Appendix.pdf](./Appendix.pdf).

---

## Repository Structure

```
├── README.md                          # This file
├── Appendix.pdf                       # Extended results, construction details, qualitative analysis
├── LICENSE                            # MIT
├── requirements.txt                   # Python dependencies
├── CONFIG/
│   ├── canonical_splits.py            # Single source of truth for 70/15/15 stratified splits (seed=42)
│   ├── requirements.txt               # Full dependency list
│   └── Dockerfile                     # Optional containerization
├── Source/
│   ├── src/
│   │   ├── canonical_splits.py        # Data split API
│   │   ├── counterfactual_gen/
│   │   │   ├── hate_speech_dataset_builder.py    # Build 6k base corpus from Kennedy et al.
│   │   │   ├── CF-Gen.py                         # LLM-guided counterfactual text generation
│   │   │   ├── utils.py                          # Utility functions
│   │   │   ├── config.py                         # Configuration
│   │   │   └── hate_speech_dataset_6k.csv        # Base dataset (6k originals)
│   │   ├── data/
│   │   │   └── [Placeholder for processed datasets]
│   │   ├── image_gen/
│   │   │   ├── generate_t2i_prompts.py           # Prompt enhancement for Z-Image-Turbo
│   │   │   └── image_gen.py                      # Z-Image-Turbo batch generation; 720×720 PNG
│   │   ├── text_models/
│   │   │   ├── binary_fairness_analysis.py       # TF-IDF (10k unigrams) + LR, SVM, Ridge, NB, RF
│   │   │   └── train_hatebert.py                 # HateBERT end-to-end fine-tuning
│   │   ├── image_models/
│   │   │   ├── image_train.py                    # EfficientNet-B0 and CLIP ViT-B/32 training
│   │   │   ├── model.py                          # Model architectures with GRL adversarial head
│   │   │   ├── grl_lambda_sweep.py               # Adversarial weight sensitivity analysis
│   │   │   └── config.py                         # Model configuration
│   │   ├── fusion/
│   │   │   ├── late_fusion_ensemble.py           # Late-fusion (equal weight, weighted average, LR-Poly)
│   │   │   ├── stacking_ensemble.py              # Stacking with LR, MLP, GBT meta-learners
│   │   │   └── cross_attention_fusion.py         # Cross-attention GMU with optional GRL
│   │   └── analysis/
│   │       ├── enhanced_statistical_tests.py     # OLS ANOVA, Chi-square, Kruskal-Wallis, Wilcoxon
│   │       ├── image_leakage_audit.py            # Verify no class-discriminative text in images
│   │       └── per_group_text_dfpr.py            # Per-group FPR analysis
│   ├── tests/
│   │   └── test_integration.py                   # Integration tests for pipeline
│   ├── scripts/
│   │   ├── generate_all_plots.py                 # Generate result visualizations
│   │   └── [Additional utility scripts]
│   └── docs/
│       ├── DATASET.md                            # Dataset construction details
│       ├── MODELS.md                             # Model architectures and hyperparameters
│       └── REPRODUCE.md                          # Step-by-step reproduction guide
└── OOD-testing/
    └── [Out-of-domain evaluation on HateXplain]
```

---

## Quickstart

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU-accelerated training)
- ~50 GB disk space (for datasets and model checkpoints)

### 1. Clone and Install

```bash
git clone https://github.com/[repo].git
cd COUNTER-HATE
pip install -r requirements.txt
```

### 2. Verify Canonical Splits

```bash
python canonical_splits.py
# Output: 5,941 total originals; 70/15/15 stratified split verified
```

### 3. Generate Counterfactuals (Optional; already included)

```bash
cd Source/src
python counterfactual_gen/CF-Gen.py \
  --input hate_speech_dataset_6k.csv \
  --output final_dataset_18k.csv \
  --model "Qwen/Qwen2.5-7B-Instruct"
# Runtime; ~4 hours on Kaggle T4×2 GPU
```

### 4. Train Text Models

```bash
cd Source/src
python text_models/binary_fairness_analysis.py \
  --dataset final_dataset_18k.csv \
  --condition CF \
  --output text_models/results/
# Models; TF-IDF + LR, SVM, Ridge, NB, Random Forest; HateBERT fine-tuning
# Results written to results/ directory
```

### 5. Train Image Models

```bash
python image_models/image_train.py \
  --condition CF+GRL \
  --epochs 50 \
  --seed 42 \
  --output image_models/results/
# Models; EfficientNet-B0, CLIP ViT-B/32 with GRL adversarial head
# Multi-seed training; seeds 42, 123, 456
```

### 6. Fuse and Evaluate

```bash
python fusion/late_fusion_ensemble.py \
  --text_results text_models/results/ \
  --image_results image_models/results/ \
  --output fusion/results/
# Fusion strategies; equal weight, learned weights, stacking (LR/MLP/GBT), LR-Poly, Cross-Attention GMU
# Best config; HateBERT CF + CLIP CF+GRL, LR(2D), w=0.50 → F1=0.884, AUC=0.968, FPR=0.225
```

### 7. Statistical Testing

```bash
python analysis/enhanced_statistical_tests.py \
  --fusion_results fusion/results/ \
  --output analysis/results/
# Tests; OLS ANOVA (condition×group interaction: F=9.82, p=1.7×10−10), Chi-square, Kruskal-Wallis, Wilcoxon
```

### 8. Out-of-Domain Evaluation

```bash
cd OOD-testing
python run_ood_evaluation.py \
  --model HateBERT \
  --condition CF \
  --dataset HateXplain
# OOD results; N=20,148; FPR reduction 0.357→0.319 (−10.5% relative)
```

---

## Models and Experimental Conditions

### Text Models

**Architectures:** TF-IDF (10,000 unigrams) with Logistic Regression, SVM (kernel=linear), Ridge regression, Naive Bayes, Random Forest (100 trees); HateBERT (berts-hate-speech-offensive/hatebert) with end-to-end fine-tuning (lr=2×10−5, epochs=5, batch=16, AdamW, warmup=0.1, max_seq=128).

**Conditions:**
- **nCF:** 4,158 training originals (6,000 total across dataset)
- **CF:** 12,474 training samples (4,158 originals + 8,316 counterfactuals)

### Image Models

**Architectures:** EfficientNet-B0 and CLIP ViT-B/32 with optional Gradient Reversal Layer (GRL) adversarial head for group-protected invariance.

**Conditions:**
- **nCF:** 6,000 originals only; no augmentation
- **CF-no-adv:** 18,000 augmented samples; no adversarial head
- **CF+GRL:** 18,000 augmented samples; GRL with λ=0.5 (balances task and adversarial loss equally)

**Hyperparameters:** AdamW (lr=1×10−4), label smoothing ε=0.05, max 50 epochs, early stopping patience=7 on validation F1, batch size 32, seeds 42/123/456, input 720×720→224×224 resize, ImageNet normalization.

### Fusion Strategies

**Late Fusion:** Averages text and image class probabilities with grid-searched weight w ∈ [0.0, 1.0]. 
Optimal: w=0.50 for nCF and CF+GRL (semantic parity), w=0.45 for CF-no-adv.

**Stacking:** 5-fold internal CV on training outputs; trains meta-learner (LR, MLP, GBT) on out-of-fold probability scores. Best; LR and GBT with isotonic regression calibration (ECE −73%; reduces from 0.18 to 0.048).

**Cross-Attention GMU:** Gated Multimodal Unit over 384-dimensional embedding concatenation; optional GRL head matching image model design.

**Calibration:** Post-hoc isotonic regression on validation set scores; applied to all test outputs without refitting. It reduces ECE by 54–82% depending on model.

---

## Experimental Conditions Explained

In this work, we evaluate three configurations for both text and image modalities;

1. **nCF (No Counterfactual):** Baseline; models trained on 6,000 original samples only. No augmentation.

2. **CF (Counterfactual, no adversarial):** Text and image models trained on full 18,000 augmented corpus. For image models, this means all 18,000 synthetically generated images are used; no adversarial debiasing at the representation level.

3. **CF+GRL (Counterfactual + Gradient Reversal Layer):** Image models augmented with adversarial fairness head that explicitly penalizes group-discriminative features via negated gradient flow. λ=0.5 balances task loss and adversarial loss equally. Text models use CF only (adversarial training not applied to text).

Text models show that CF alone provides substantial FPR reduction (−12.3% average per-group FPR). Image models show that CF without adversarial debiasing introduces new visual biases; GRL training mitigates these by 10–15% additional reduction in equalized odds disparity.

---
## Key Results

| Configuration | Modality | Model | F1 | AUC | FPR | Notes |
|---|---|---|---|---|---|---|
| nCF (baseline) | Text | HateBERT | 0.856 | 0.927 | 0.203 | 6k original samples only |
| CF (augmented) | Text | HateBERT | 0.873 | 0.941 | 0.178 | 18k training samples; per-group ΔFPR −12.3% |
| CF+GRL | Image | CLIP ViT-B/32 | 0.841 | 0.891 | 0.286 | Gradient Reversal Layer; multi-seed stable |
| **CF+GRL Fusion** | **Multimodal** | **HateBERT + CLIP (LR 2D)** | **0.884** | **0.968** | **0.225** | **Best overall; isotonic ECE=0.020** |
| OOD (HateXplain) | Text | HateBERT CF | 0.540 | 0.719 | 0.319 | Distribution shift; FPR reduction 0.357→0.319 (−10.5%) |

---
## Citation

[Citation placeholder; paper under review at ACM MM 2026]
---

## License

MIT License; see [LICENSE](LICENSE) for details.

## Questions?

For reproducibility questions or issues; consult [Source/docs/REPRODUCE.md](Source/docs/REPRODUCE.md) or the discussion section.
