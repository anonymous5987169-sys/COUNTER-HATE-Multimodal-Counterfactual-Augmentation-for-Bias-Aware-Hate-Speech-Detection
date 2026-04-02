# REPRODUCE: Step-by-Step Reproduction Guide

This guide provides detailed instructions to reproduce all results reported in the COUNTER-HATE paper.

**Estimated runtime;** Varies by stage (see section runtimes below). Full end-to-end pipeline; ~8 days on H200 GPU for image generation + 3 days for all text and image model training.

---

## Prerequisites

### System Requirements

- Python 3.9 or higher
- CUDA 11.8+ (for GPU acceleration; optional but recommended)
- 50+ GB available disk space (for datasets; models; checkpoints; image outputs)
- GPU with 16+ GB VRAM (for EfficientNet; CLIP; HateBERT fine-tuning)

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages;

- torch; transformers; scikit-learn; pandas; numpy
- EfficientNet; LoRA; timm
- ComfyUI (for image generation; optional)

---

## Stage 1: Verify Canonical Data Splits

**Runtime;** < 1 minute

**Purpose;** Confirm the 70/15/15 stratified split at original_sample_id level.

```bash
cd Source/src

python canonical_splits.py
```

**Expected output;**

```
✓ Canonical splits verified
  Total originals: 5,941
  Train: 4,158 (69.99%)
  Val: 891 (15.00%)
  Test: 892 (15.01%)
  Random seed: 42
```

**Output file;** `canonical_splits.json` contains the split assignments (keyed on original_sample_id).

---

## Stage 2: Generate Counterfactuals from Base Corpus

**Runtime;** ~4 hours on Kaggle T4×2 GPU; ~2 days on CPU

**Purpose;** Generate 12;000 counterfactual text variants from 6;000 originals using Qwen2.5-7B-Instruct.

### 2a. Prepare Base Corpus

```bash
cd Source/src/counterfactual_gen

# If base CSV not present; build from HuggingFace
python hate_speech_dataset_builder.py \
  --output hate_speech_dataset_6k.csv \
  --seed 42
```

**Expected;** CSV with 6;000 rows; columns [text; class_label; target_group; polarity; sample_id]

### 2b. Generate Counterfactuals

```bash
python CF-Gen.py \
  --input hate_speech_dataset_6k.csv \
  --output final_dataset_18k.csv \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --batch_size 32 \
  --device cuda  # or cpu
```

**Key parameters;**
- `--temperature 0.25`: Fixed for reproducibility
- `--top_p 0.9`: Nucleus sampling
- `--max_tokens 128`: Max length per variant
- `--seed 42`: For determinism (seeds per batch; though stochastic generation)

**Expected output;** `final_dataset_18k.csv` with 18;000 rows; columns [text; class_label; target_group; original_sample_id; variant_id]

**Validation;** Script includes CJK guard and fallback substitution; ~2% of samples expected to use deterministic fallback

---

## Stage 3: Generate Synthetic Images

**Runtime;** ~39 hours on H200 GPU

**Purpose;** Generate 18;000 synthetic 720×720 images from text samples via Z-Image-Turbo.

### 3a. Enhance Prompts

```bash
cd Source/src/image_gen

python generate_t2i_prompts.py \
  --input ../counterfactual_gen/final_dataset_18k.csv \
  --output final_dataset_18k_with_prompts.csv
```

**Expected;** CSV with added column `image_prompt` containing structured; multi-sentence visual descriptions

### 3b. Generate Images (Batch Mode)

```bash
python image_gen.py \
  --input final_dataset_18k_with_prompts.csv \
  --output_dir ./generated_images \
  --model "Z-Image-Turbo" \
  --resolution 720 \
  --diffusion_steps 9 \
  --cfg_scale 1.0 \
  --batch_size 256 \
  --device cuda \
  --seed_base 0xDEADBEEF \
  --seed_prime 1000003 \
  --checkpoint_interval 5
```

**Key parameters;**
- `--deterministic_seed`: Formula (SEED_BASE + i × SEED_PRIME) mod 2^32 ensures reproducibility
- `--checkpoint_interval 5`: Flush progress every 5 batches; allows resumption
- `--batch_size 256`: Adaptive halving on OOM

**Expected output;** `generated_images/` directory with 18;000 PNG files organized by binary label;

```
generated_images/
  ├── hate/
  │   ├── 0.png
  │   ├── 1.png
  │   └── ... (9;000 files)
  └── non_hate/
      ├── 0.png
      ├── 1.png
      └── ... (9;000 files)
```

**Resumption;** If interrupted; script resumes from checkpoint and skips already-generated images.

---

## Stage 4: Train Text Models

**Runtime;** ~6 hours for all baselines + HateBERT on GPU; ~3 days on CPU

### 4a. TF-IDF Baselines (nCF and CF conditions)

```bash
cd Source/src/text_models

python binary_fairness_analysis.py \
  --dataset ../counterfactual_gen/final_dataset_18k.csv \
  --condition nCF \
  --test_set canonical \
  --seed 42 \
  --output ./results_ncf/ \
  --models "lr,svm,ridge,rf,nb"
```

**Conditions;** nCF (originals only); CF (full augmented corpus)

```bash
python binary_fairness_analysis.py \
  --dataset ../counterfactual_gen/final_dataset_18k.csv \
  --condition CF \
  --test_set canonical \
  --seed 42 \
  --output ./results_cf/
```

**Expected outputs;**
- Per-model predictions; `predictions_{model}.csv`
- Per-model metrics; `metrics_{model}.json` (F1; AUC; FPR; per-group FPR)
- Aggregated report; `binary_fairness_results.json`

### 4b. HateBERT End-to-End Fine-Tuning

```bash
python train_hatebert.py \
  --dataset ../counterfactual_gen/final_dataset_18k.csv \
  --condition CF \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --warmup 0.1 \
  --seed 42 \
  --output ./hatebert_results/ \
  --device cuda
```

**Key hyperparameters;** Match Appendix B exactly (lr=2e-5; epochs=5; batch=16; max_seq=128)

**Expected outputs;**
- Fine-tuned model checkpoint; `hatebert_cf_seed42.pt`
- Predictions on test set; `hatebert_cf_test_preds.csv`
- Metrics; `hatebert_cf_metrics.json`

**Stability;** Bootstrap 1;500 resamples to compute 95% CI on test F1.

---

## Stage 5: Train Image Models

**Runtime;** ~2 days for all seeds + conditions on GPU

**Note;** EfficientNet-B0 and CLIP ViT-B/32 trained in parallel across 3 random seeds (42; 123; 456).

### 5a. EfficientNet-B0

```bash
cd Source/src/image_models

python image_train.py \
  --model efficientnet \
  --image_dir ../image_gen/generated_images/ \
  --dataset ../counterfactual_gen/final_dataset_18k.csv \
  --conditions "nCF,CF-no-adv,CF+GRL" \
  --seeds "42,123,456" \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --early_stopping_patience 7 \
  --output ./efficientnet_results/
```

**Expected outputs;** (per seed; per condition)
- Checkpoint; `efficientnet_condition_seed.pt`
- Metrics; `efficientnet_condition_seed_metrics.json` (F1; AUC; FPR; ECE raw)
- Multi-seed summary; `efficientnet_multiseed_summary.json`

### 5b. CLIP ViT-B/32

```bash
python image_train.py \
  --model clip \
  --image_dir ../image_gen/generated_images/ \
  --dataset ../counterfactual_gen/final_dataset_18k.csv \
  --conditions "nCF,CF-no-adv,CF+GRL" \
  --seeds "42,123,456" \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --early_stopping_patience 7 \
  --output ./clip_results/
```

**Expected outputs;** (per seed; per condition)
- Checkpoint; `clip_cf_grl_seed.pt`
- Metrics; `clip_cf_grl_seed42_metrics.json`
- Frozen backbone flag; freeze_encoder=True (CLIP encoder frozen)

### 5c. Adversarial Weight Sensitivity (λ sweep)

```bash
python grl_lambda_sweep.py \
  --model clip \
  --image_dir ../image_gen/generated_images/ \
  --dataset ../counterfactual_gen/final_dataset_18k.csv \
  --lambda_values "0.1,0.3,0.5,0.7,1.0" \
  --output ./lambda_sweep_results/
```

**Expected;** Report showing F1 and FPR across λ values; confirms λ=0.5 is near-optimal.

---

## Stage 6: Multimodal Fusion

**Runtime;** ~2 hours for all strategies with 5-fold CV

### 6a. Late-Fusion (Learned Weights)

```bash
cd Source/src/fusion

python late_fusion_ensemble.py \
  --text_probs ../text_models/hatebert_cf_test_preds.csv \
  --image_probs ../image_models/clip_cf_grl_seed42_test_preds.csv \
  --strategy "weighted_average" \
  --weight_grid "0.0:0.1:1.0" \
  --output ./late_fusion_results/ \
  --calibration isotonic
```

**Grid search;** $w \in \{0.0; 0.1; ...; 1.0\}$ on validation set

**Expected output;** Best weight ($w = 0.50$); F1; AUC; FPR on test set

### 6b. Stacking (Meta-Learner)

```bash
python stacking_ensemble.py \
  --text_probs ../text_models/hatebert_cf_test_preds.csv \
  --image_probs ../image_models/clip_cf_grl_seed42_test_preds.csv \
  --cv_folds 5 \
  --meta_learners "lr,mlp,gbt" \
  --output ./stacking_results/ \
  --calibration isotonic
```

**Expected;** Per-metalearner metrics; best;  LR(2D polynomial)

### 6c. Cross-Attention Fusion

```bash
python cross_attention_fusion.py \
  --text_embeddings ../text_models/hatebert_cf_embeddings.npy \
  --image_embeddings ../image_models/clip_cf_grl_seed42_embeddings.npy \
  --cv_folds 5 \
  --use_grl True \
  --output ./cross_attention_results/
```

**Expected;** F1; AUC; ECE across 5-fold CV

---

## Stage 7: Statistical Testing

**Runtime;** ~30 minutes

```bash
cd Source/src/analysis

python enhanced_statistical_tests.py \
  --fusion_results ../fusion/late_fusion_results/metrics.json \
  --text_results ../text_models/binary_fairness_results.json \
  --per_group_fpr ../text_models/hatebert_cf_per_group_fpr.csv \
  --alpha 0.05 \
  --output ./statistical_results.json
```

**Tests reported;**
- OLS ANOVA (condition × group interaction; F=9.82; p=1.7×10^−10)
- Chi-square (per-group FPR disparity)
- Kruskal-Wallis (non-parametric alternative)
- Wilcoxon signed-rank (seed-level comparison for image models)

---

## Stage 8: Out-of-Domain Evaluation (HateXplain)

**Runtime;** ~1 hour

```bash
cd Source/OOD-testing

python run_ood_evaluation.py \
  --model HateBERT \
  --checkpoint ../src/text_models/hatebert_cf_seed42.pt \
  --dataset HateXplain \
  --condition CF \
  --output ./ood_results.json
```

**Expected;** F1=0.540; AUC=0.719; FPR=0.319 (10.5% relative reduction vs. nCF baseline)

---

## Stage 9: Verification and Leakage Audit

### 9a. Image Leakage Audit

```bash
cd Source/src/analysis

python image_leakage_audit.py \
  --image_dir ../image_gen/generated_images/ \
  --sample_size 200 \
  --output ./leakage_audit_report.json
```

**Expected;** 0 readable class-discriminative text detected; confirms anti-text negatives effective

### 9b. Per-Group FPR Analysis

```bash
python per_group_text_dfpr.py \
  --predictions ../text_models/hatebert_cf_test_preds.csv \
  --dataset ../counterfactual_gen/final_dataset_18k.csv \
  --output ./per_group_fpr.csv
```

**Expected;** Per-group FPR reductions; Sexual Orientation (−18.2%); Age (−6.3%)

---

## Troubleshooting

### Common Issues

**KeyError; original_sample_id not found**
- Check that canonical_splits.py has been run; canonical_splits.json must exist

**CUDA OOM during image generation**
- Reduce batch_size from 256 to 128; script has automatic halving

**HateBERT model not found**
- Ensure transformers library can download from HuggingFace; check internet connection

**Mismatched test set**
- Verify test set always uses 892 originals only; no counterfactuals

### Validation Commands

```bash
# Verify 18k dataset integrity
python -c "import pandas as pd; df=pd.read_csv('final_dataset_18k.csv'); print(f'Rows: {len(df)}; Cols: {list(df.columns)}')"

# Check image count
ls generated_images/hate/*.png | wc -l  # expect 9000
ls generated_images/non_hate/*.png | wc -l  # expect 9000

# Verify canonical splits
python canonical_splits.py --verify
```

---

## Expected Final Results

If all stages complete successfully;

| Config | F1 | AUC | FPR |
|---|---|---|---|
| HateBERT CF (text only) | 0.873 | 0.941 | 0.178 |
| CLIP CF+GRL (image only) | 0.841 | 0.891 | 0.286 |
| Fusion (HateBERT + CLIP; LR 2D) | **0.884** | **0.968** | **0.225** |
| OOD HateXplain (HateBERT CF) | 0.540 | 0.719 | 0.319 |

Reproducibility tolerance; ±0.005 on F1; ±0.003 on FPR due to multi-seed variation.

---

## Questions?

Refer to;
- [Source/docs/DATASET.md](../docs/DATASET.md) for dataset details
- [Source/docs/MODELS.md](../docs/MODELS.md) for model specifications
- Individual script docstrings (e.g.; `python script.py --help`)
