# CLIP Scoring Pipeline

Complete pipeline for computing CLIP similarity scores between image-text pairs in the 18K hate speech dataset with counterfactual augmentation.

## Overview

This pipeline:
1. Builds metadata mapping images to their corresponding texts and split types (original vs counterfactual)
2. Computes OpenAI CLIP-ViT-Base-32 similarity scores for each (image, text) pair
3. Generates summary statistics grouped by split type and class label

## Requirements

Install dependencies:
```bash
pip install -r requirements_clip.txt
```

Or manually install:
```bash
pip install torch torchvision transformers pillow pandas tqdm numpy
```

## Quick Start

### Full 18K Dataset (6000 original + 12000 counterfactual)

```bash
# Step 1: Build metadata mapping images to texts
python build_metadata.py

# Step 2: Compute CLIP scores for all 18K pairs
python compute_clip_scores.py --device cuda

# Step 3: Generate summary statistics
python summarize_clip.py
```

### Testing with Small Sample (20 per class)

```bash
# Build test metadata with 20 images per class (160 total)
python build_metadata.py --limit-per-class 20

# Compute scores on test set
python compute_clip_scores.py --metadata metadata.csv --output clip_scores_test.csv --device cuda

# Summarize test results
python summarize_clip.py --input clip_scores_test.csv --output clip_summary_test.csv
```

## Scripts

### 1. `build_metadata.py`

Builds `metadata.csv` that maps image files to their corresponding texts and split types.

**Usage:**
```bash
python build_metadata.py [--output metadata.csv] [--limit-per-class N]
```

**Options:**
- `--output`: Output CSV filename (relative to `clip_scoring/`)
- `--limit-per-class`: Limit to N samples per class (for testing); if None, uses all samples

**Output:** CSV with columns:
- `image_path`: Relative path to image file
- `text`: Text description of image
- `split_type`: "original" or "counterfactual"
- `class_label`: Hate speech class (8 classes)
- `counterfactual_id`: Unique identifier matching CSV row

### 2. `compute_clip_scores.py`

Computes CLIP similarity scores for each (image, text) pair.

**Usage:**
```bash
python compute_clip_scores.py \
  --metadata metadata.csv \
  --output clip_scores_results.csv \
  --device cuda \
  --batch-size 32
```

**Options:**
- `--metadata`: Input metadata CSV (relative to `clip_scoring/`)
- `--output`: Output CSV with clip_score column
- `--device`: Device to use (`cuda` or `cpu`)
- `--batch-size`: Batch size for processing (default: 32)
- `--use-fp16`: Use float16 precision (not recommended)

**Output:** CSV with input columns + `clip_score` column
- `clip_score`: CLIP similarity score in range [0, 1]

**Features:**
- CUDA support with automatic CPU fallback
- Progress bar with tqdm
- Automatic text truncation to 77 tokens (CLIP limit)
- Prints summary statistics grouped by split_type and class_label

### 3. `summarize_clip.py`

Generates detailed summary statistics from results CSV.

**Usage:**
```bash
python summarize_clip.py \
  --input clip_scores_results.csv \
  --output clip_summary_table.csv
```

**Options:**
- `--input`: Input results CSV (relative to `clip_scoring/`)
- `--output`: Output summary CSV (relative to `clip_scoring/`)

**Output:** 
- Prints detailed statistics to console
- Saves grouped summary to CSV with statistics per split_type:
  - `split_type`, `count`, `mean`, `std`, `min`, `q25`, `median`, `q75`, `max`

## File Structure

```
clip_scoring/
├── requirements_clip.txt              # Python dependencies
├── build_metadata.py                  # Metadata builder
├── compute_clip_scores.py             # CLIP scoring engine
├── summarize_clip.py                  # Summary statistics generator
├── analyze_text_quality.py            # Text quality analysis tool
├── test_clip_fix.py                   # Quick test script
├── optimized_clip_scoring.py          # Optimized scoring with temperature tuning
├── debug_clip_api.py                  # Debug utility for CLIP API
├── results/                           # Output directory
│   ├── metadata.csv                   # Image-text mapping (18K rows)
│   ├── clip_scores_results.csv        # Results: images + texts + CLIP scores
│   └── clip_summary_table.csv         # Summary statistics by split_type
└── README.md                          # This file
```

## Image Directory Structure

Images are located in:
- **Hate speech**: `Hate/{Hate_race|Hate_religion|Hate_gender|Hate_Others}/generated_images/`
- **Non-hate**: `non-hate/generated_images-{offensive-non-hate|neutral|counter-speech|ambigious}/`

Each image is named following the pattern:
- Original: `{CLASS}_NNNN.png` (e.g., `HS_HATERACE_0000.png`)
- Counterfactual CF1: `{CLASS}_NNNN_cf1.png`
- Counterfactual CF2: `{CLASS}_NNNN_cf2.png`

## Dataset Information

### Split Distribution
- **Original samples**: 6,000 (1 image per text; no duplicates in val/test)
- **Counterfactual samples**: 12,000 (2 additional CFs per original; train only)
- **Total images**: 18,000 (6,000 originals × 3 variants each)

### Class Labels (8 classes, 750 originals each)
1. `hate_race` - Hate speech targeting race/ethnicity
2. `hate_religion` - Hate speech targeting religion
3. `hate_gender` - Hate speech targeting gender
4. `hate_other` - Other hate speech
5. `offensive_non_hate` - Offensive but not hateful
6. `neutral_discussion` - Neutral discussions
7. `counter_speech` - Counter-hate speech
8. `ambiguous` - Ambiguous class

## Performance Notes

- **GPU Required**: Strongly recommended for speed (NVIDIA 3050 tested)
  - Full 18K dataset: ~20-30 minutes on CUDA
  - Test 160 samples: ~10 seconds on CUDA
  
- **Memory Usage**:
  - CLIP model: ~700 MB on CUDA
  - Batch processing: minimal additional memory overhead with batch_size=32

- **Device Fallback**: 
  - Automatic CPU fallback if CUDA unavailable
  - CPU processing ~50x slower (use for testing only)

## Results Summary (Full 18K Dataset)

Complete CLIP scoring results with comprehensive statistics are provided in the results/ directory.

Key metrics:
- **Total Samples**: 18,000
- **Model**: openai/clip-vit-base-patch32 (ViT-B/32)
- **Device**: CUDA GPU
- **Output File**: `clip_scores_results.csv`

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size` (e.g., 16 or 8)
- Use `--device cpu` for testing

### Text Token Limit Errors
- Automatically handled by script (truncation to 77 tokens)
- No manual intervention needed

### Missing Image Files
- Verify image directories exist at `Hate/*/generated_images/` and `non-hate/generated_images-*/`
- Check file naming follows pattern: `{CLASS}_NNNN.png`, `{CLASS}_NNNN_cf1.png`, etc.

## Citation

If using this pipeline in research, cite:
- OpenAI CLIP: Radford et al. (2021) "Learning Transferable Models for Vision Tasks"
- Dataset: Original work on counterfactual data augmentation for hate speech detection
