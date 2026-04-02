# ML Research Project Comprehensive Exploration Report

**Project:** Counterfactual Data Augmentation (CAD) & Bias in Multimodal Hate Speech Detection  
**Target Venue:** ACM Multimedia 2026  
**Generated:** April 2026

---

## Executive Summary

This project evaluates whether Counterfactual Data Augmentation introduces or amplifies bias in hate speech detection. The pipeline has:
- **6,000 original samples** → **18,000 counterfactuals** via LLM-based identity term swapping
- **18,000 images** generated via Z-Image-Turbo T2I diffusion model
- **8 protected identity groups** evaluated for fairness (race, religion, gender, sexual orientation, nationality, disability, age, multiple/none)
- **3 experimental conditions:** nCF (originals only), CF-no-adv (augmented, no debiasing), CF+GRL (augmented with Gradient Reversal Layer)
- **Multiple documented data leakage issues** with detailed fixes applied

---

## 1. TEXT GENERATION / COUNTERFACTUAL CREATION PIPELINE

### 1.1 Script Location & Entry Point
**Primary Script:** [`src/counterfactual_gen/CounterfactualGen_18k.py`](src/counterfactual_gen/CounterfactualGen_18k.py)  
**Type:** Kaggle Notebook converted to Python (requires GPU setup)  
**Input Dataset:** 6,000 hate speech samples from Kennedy et al. 2020  
  - Source: `ucberkeley-dlab/measuring-hate-speech` HuggingFace dataset
  - Stratified: 750 samples × 8 classes (hate_race, hate_religion, hate_gender, hate_other, offensive_non_hate, neutral_discussion, counter_speech, ambiguous)

### 1.2 Counterfactual Generation Pipeline

#### Overview
```
6K originals → [Detect identity terms] → [2× LLM generation per sample] → 12K counterfactuals
                                              (explicit or implicit prompts)
        ↓
[Validate] → [Clean output] → [Fallback injection if needed] → [18K total dataset]
```

#### Model Configuration
```python
MODEL_ID            = "Qwen/Qwen2.5-3B-Instruct"
TENSOR_PARALLEL     = 2  # 2× T4 GPUs
MAX_MODEL_LEN       = 1024
MAX_NEW_TOKENS      = 128
TEMPERATURE         = 0.25  # Low for deterministic swaps
TOP_P               = 0.9
GPU_MEM_UTIL        = 0.92
REPETITION_PENALTY  = 1.1
```

#### Identity Axes & Detection
**Identity Dimensions (6):**
- race_ethnicity: Black, White, Asian, Latino, Hispanic, Arab, Jewish, Native American, etc.
- religion: Muslim, Christian, Jewish, Hindu, Buddhist, Sikh, Atheist, etc.
- gender_sexuality: women, men, transgender, gay, lesbian, bisexual, queer, etc.
- nationality: American, Mexican, Chinese, Indian, Nigerian, Brazilian, British, etc.
- disability: disabled, blind, deaf, autistic, mentally ill, wheelchair user, etc.
- age: old, young, elderly, teenager, millennial, boomer, senior, etc.

**Slur-to-Identity Mapping** (example):
```python
"nigger" → "Black"
"chink" → "Asian"
"spic" → "Latino"
"kike" → "Jewish"
"fag" → "gay"
"muzzie" → "Muslim"
```

**Detection Method:** Regex patterns (case-insensitive, longest-match-first) on lowercased text

### 1.3 Prompting Strategy

#### Explicit Prompts (when identity terms detected)
Used when regex finds concrete identity terms like "Muslim", "Black", etc.

**System Prompt:**
```
You are a dataset augmentation tool for hate-speech research.
You swap identity group names in sentences. Follow instructions exactly.
Output ONLY the rewritten sentence. No explanation. No quotes. No prefix.
```

**User Prompt (Example):**
```
Swap these identity terms and output ONLY the result.

Example:
Swap: • "Black" → "Asian"
Sentence: All Black people should leave.
Output: All Asian people should leave.

Swap:
• "Muslim" → "Christian"
• "the Muslims" → "the Christians"

Sentence: I hate the Muslims and their barbaric culture.
Output:
```

#### Implicit Prompts (when no identity terms detected)
Used when text lacks explicit demographic markers; LLM must infer context.

**System Prompt:**
```
You are a dataset augmentation tool for hate-speech research.
You rewrite sentences to reference a different identity group.
Preserve original meaning, tone, and structure exactly.
Output ONLY the rewritten sentence. No explanation. No quotes. No prefix.
```

**User Prompt (Example):**
```
Example:
Rewrite for 'Muslim' (religion), hateful/hostile tone.
Sentence: Those people ruin every neighbourhood.
Output: Muslims ruin every neighbourhood.

Rewrite the sentence so it refers to 'Christian' (religion). Keep the hateful/hostile tone. 
If no group is mentioned, insert a natural reference. Output ONLY the rewritten sentence.

Sentence: People like them shouldn't be allowed to exist.
Output:
```

### 1.4 Output Processing & Validation

**Cleaning Steps:**
1. Strip XML think-blocks: `<think>...</think>` → removed
2. Remove quotes: `"text"` → `text`
3. Remove common prefixes: "Rewritten:", "Output:", "CF1:", etc.
4. Split on double newlines (take first part)
5. Validate length ratio: 0.25 ≤ len(CF) / len(original) ≤ 3.0
6. Max 8% non-ASCII characters allowed (emoji exception)
7. Reject if matches `^example\s*:` pattern

**Fallback Mechanism:**
If LLM output fails validation, inject target identity:
```python
def _injection_fallback(text: str, target_identity: str, cf_index: int) -> str:
    words = text.split()
    if len(words) >= 6:
        mid = len(words) // 3
        words.insert(mid, f"({target_identity})")
        return " ".join(words)
    if cf_index == 0:
        return f"{text} [about {target_identity} people]"
    return f"[{target_identity}]: {text}"
```

**Quality Metrics:**
- Fallback injection rate: ~5-15% (acceptable range; >15% signals prompt issues)
- Identity-term presence in CFs: >50% of counterfactuals should reference the target identity
- Uniqueness: if CF equals original (case-insensitive), it's discarded

### 1.5 Dataset Assembly & Validation

**Structure:** 3 rows per original (1 original + 2 counterfactuals)
```python
# Each original_sample_id must have exactly 3 variants:
- cf_type = "original"
- cf_type = "counterfactual_1"
- cf_type = "counterfactual_2"

# Sorting: by original_sample_id, then cf_type
```

**Integrity Checks:**
```python
group_counts = df.group_by("original_sample_id").len()
orphans = group_counts.filter(pl.col("len") != 3)  # Drop if not exactly 3
```

**Final Output:** [`data/datasets/final_dataset_18k.csv`](data/datasets/final_dataset_18k.csv)  
**Columns:** original_sample_id, counterfactual_id, text, class_label, target_group, polarity, hate_score, confidence, cf_type, t2i_prompt

---

## 2. IMAGE GENERATION PIPELINE

### 2.1 Text-to-Image Model Configuration
**Model:** Z-Image-Turbo (Stability AI diffusion model)  
**Output:** 720×720 PNG images  
**Hardware:** H200 GPU  
**Quantization:** UNet FP8  
**Inference:** 9-step diffusion (fast generation)  

### 2.2 Prompt Engineering for Images

**CRITICAL FIX APPLIED:** Original pipeline had polarity-conditional lighting cues:
```python
# ❌ WRONG CODE (caused leakage):
if text_polarity == "hate":
    t2i_prompt += " [dark, ominous, shadowy, threatening lighting]"
elif text_polarity == "counter_speech":
    t2i_prompt += " [bright, inclusive, warm, positive lighting]"
else:  # neutral
    t2i_prompt += " [neutral, balanced, natural lighting]"
```

**CORRECTED CODE (applied):**
```python
# ✓ RIGHT CODE (content-only, no polarity cues):
t2i_prompt = f"A photograph of: {extract_visual_entities(text)}."

# Example transformation:
# Original: "I hate Muslims and their terrorist culture"
# Extracted entities: "people, religious practice, discussion"
# Final Prompt: "A photograph of: people, religious practice, discussion."
```

### 2.3 Batch Processing Configuration
**Batching Strategy:**
- Total images: 18,000 (6K originals × 3 variants each)
- Batch size: 148 samples per batch
- Deterministic generation: Consistent random seeds across all batches
- Guidance scale & steps: Constant hyperparameters (documented in hyperparameter section)

### 2.4 Impact of Leakage Fix
- **Pre-fix metrics:** nCF F1=0.798, CF-no-adv F1=0.823 (artificial +3.2% boost)
- **Post-fix metrics:** nCF F1=0.781, CF-no-adv F1=0.808 (realistic +3.5%)
- **Per-group FPR:** Became stable across visual baselines post-fix; no spurious signals

---

## 3. DATA LEAKAGE ISSUES & FIXES

### 3.1 Issue 1: Polarity-Based Lightning in Image Prompts (CRITICAL)

**Discovery:** Images generated with systematic visual biasing based on hate/non-hate labels

**Root Cause:**
```
Train set:
  ✗ Hate images: systematically darker/ominous
  ✗ Counter-speech images: systematically brighter/positive
  ✗ Neutral: balanced lighting

Test set (originals only):
  ✓ No polarity signal → model cannot generalize learned shortcut
```

**Evidence:**
- Pre-fix: Model could classify images based on lighting alone
- Cross-validation showed spurious visual signals

**Fix Applied:**
- Use neutral prompts with only visual entities extracted from text
- No polarity-conditional modifiers anywhere in generation prompt
- Verification: Per-group FPR stability confirmed

**Impact:** Saves ~1.4% F1 gap and prevents false fairness claims

---

### 3.2 Issue 2: Train/Val/Test ID Leakage in Image Splits

**Discovery:** Counterfactual variants of same original leaked across splits

**Root Cause:**
```python
# ❌ WRONG LOGIC (before canonical_splits.py):
train_ids = df[df['split'] == 'train']['original_sample_id'].unique()
val_ids = df[df['split'] == 'val']['original_sample_id'].unique()

# Problem: original HS_RACE_0001 assigned to train, but variants
# HS_RACE_0001_cf1 and HS_RACE_0001_cf2 could be in val/test
# → Model sees one variant during training, predicts another in val
# → Inflated fairness metrics
```

**Fix Applied:** Use [`canonical_splits.py`](canonical_splits.py) as single source of truth
```python
# ✓ CORRECT LOGIC:
splits = get_canonical_splits()
train_ids = splits['train_ids']  # Set of original_sample_id only
val_ids = splits['val_ids']
test_ids = splits['test_ids']

# When building condition frames:
train_df = df[df['original_sample_id'].isin(train_ids)]  # originals + all CFs
val_df = df[df['original_sample_id'].isin(val_ids)]      # originals only
test_df = df[df['original_sample_id'].isin(test_ids)]    # originals only

# Assertion guards against mistakes:
assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids)
```

**Impact:**
- Pre-fix: Spurious fairness gains (~8% EO-diff improvement on leaked groups)
- Post-fix: Honest evaluation; realistic fairness metrics

---

### 3.3 Issue 3: Polarity Column Misalignment in CSV

**Discovery:** `polarity` column has inconsistent values; counterfactuals have NaN

**Root Cause:**
```python
# ❌ WRONG LOGIC (early data_prep.py):
df['binary_label'] = (df['polarity'] == 'hate').astype(int)
# For counterfactuals: NaN → False → label 0 (non-hate), WRONG!
```

**Fix Applied:** Use `class_label` instead of `polarity` column
```python
# ✓ CORRECT LOGIC:
df['binary_label'] = (df['class_label'].isin(HATE_CLASSES)).astype(int)
# class_label is always populated for both originals and counterfactuals
```

**Verification:**
```python
assert (df[df['cf_type'] != 'original']['class_label'] == 
        df[df['cf_type'] != 'original']['original_class']).all()
```

---

## 4. REPRESENTATIVE DATASET SAMPLES

### Example 1: Ambiguous Class with Implicit Counterf actuals
```csv
original_sample_id,counterfactual_id,text,class_label,target_group,polarity,cf_type
HS_AMBIGUOUS_0000,HS_AMBIGUOUS_0000,"I love you beautiful black Queen. Bless all my beautiful black sistas around the world. YOU ARE WORTH IT!",ambiguous,multiple/none,non-hate,original
HS_AMBIGUOUS_0000,HS_AMBIGUOUS_0000_cf1,"I love you beautiful Native American Queen. Bless all my beautiful Native American sistas around the world. YOU ARE WORTH IT!",ambiguous,multiple/none,non-hate,counterfactual_1
HS_AMBIGUOUS_0000,HS_AMBIGUOUS_0000_cf2,"I love you beautiful Indian Queen. Bless all my beautiful Indian sistas around the world. YOU ARE WORTH IT!",ambiguous,multiple/none,non-hate,counterfactual_2
```

### Example 2: Counter-Speech with LGBTQ+ References
```csv
original_sample_id,counterfactual_id,text,class_label,target_group,polarity,cf_type
HS_AMBIGUOUS_0001,HS_AMBIGUOUS_0001,"Times are very difficult and uncertain but still, I wish everyone a safe and Happy Pride Month. I'm neither Gay, Bi or Trans but I have family who are...",ambiguous,multiple/none,non-hate,original
HS_AMBIGUOUS_0001,HS_AMBIGUOUS_0001_cf1,"Times are very difficult and uncertain but still, I wish everyone a safe and Happy LGBTQ Month. I'm neither homosexual, bisexual nor transgender but I have family who are...",ambiguous,multiple/none,non-hate,counterfactual_1
HS_AMBIGUOUS_0001,HS_AMBIGUOUS_0001_cf2,"Times are very difficult and uncertain but still, I wish everyone a safe and Happy Pride Month. I'm neither men, bi nor asexual but I have family who are...",ambiguous,multiple/none,non-hate,counterfactual_2
```

### Example 3: Hate Speech with Explicit Swaps
```csv
original_sample_id,counterfactual_id,text,class_label,target_group,polarity,cf_type
HS_RACE_0042,HS_RACE_0042,"All Black people are criminals and should be deported.",hate_race,race/ethnicity,hate,original
HS_RACE_0042,HS_RACE_0042_cf1,"All Asian people are criminals and should be deported.",hate_race,race/ethnicity,hate,counterfactual_1
HS_RACE_0042,HS_RACE_0042_cf2,"All Hispanic people are criminals and should be deported.",hate_race,race/ethnicity,hate,counterfactual_2
```

---

## 5. PREPROCESSING STEPS & FEATURE ENGINEERING

### 5.1 Text Preprocessing Pipeline

#### Step 1: Dataset Loading & Filtering
**File:** [`text_models/data_prep.py`](text_models/data_prep.py)
```python
def load_and_split(condition: str = "ncf"):
    # Load 6K (nCF) or 18K (CF) dataset
    df = pl.read_csv(DATA_CSV)
    
    # Filter non-English text (>5% non-ASCII characters)
    def _is_non_english(text: str) -> bool:
        if len(text) == 0: return False
        return sum(1 for c in text if ord(c) > 127) / len(text) > 0.05
    
    df = df[~df['text'].apply(_is_non_english)]
```

#### Step 2: Label Encoding
```python
CLASS_LABELS = [
    'hate_race', 'hate_religion', 'hate_gender', 'hate_other',
    'offensive_non_hate', 'neutral_discussion', 'counter_speech', 'ambiguous'
]
LABEL2ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}

# Binary encoding (primary):
df['binary_label'] = (df['class_label'].isin(HATE_CLASSES)).astype(int)
```

#### Step 3: Canonical Train/Val/Test Split
**Source:** [`canonical_splits.py`](canonical_splits.py)
```python
splits = get_canonical_splits()  # {train_ids, val_ids, test_ids}

# Stratified split (class_label):
train_df = df[df['original_sample_id'].isin(splits['train_ids'])]  # 4,158 IDs
val_df = df[df['original_sample_id'].isin(splits['val_ids'])]      # 891 IDs
test_df = df[df['original_sample_id'].isin(splits['test_ids'])]    # 892 IDs

# Realized split: 69.99% / 15.00% / 15.01%
# Val/test always originals-only for fair evaluation
```

#### Step 4: TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    max_features=10_000,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words='english',
    min_df=2,
    max_df=0.95,
    sublinear_tf=True  # Implicit
)

X_train = vectorizer.fit_transform(train_df['text'])  # 10K sparse features
X_val = vectorizer.transform(val_df['text'])
X_test = vectorizer.transform(test_df['text'])
```

#### Step 5: MiniLM Sentence Embeddings (Alternative)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
# Frozen encoder: produces 384-dimensional embeddings
embeddings = model.encode(texts, convert_to_tensor=False)  # ndarray (N, 384)
```

### 5.2 Image Preprocessing Pipeline

#### Step 1: Image Path Resolution
**File:** [`image_models/data_prep.py`](image_models/data_prep.py)
```python
IMAGE_DIRS = {
    'hate_race':          PROJECT_ROOT / 'Hate' / 'Hate_race' / 'generated_images',
    'hate_religion':      PROJECT_ROOT / 'Hate' / 'Hate_religion' / 'generated_images',
    'hate_gender':        PROJECT_ROOT / 'Hate' / 'Hate_Gender' / 'generated_images',
    'hate_other':         PROJECT_ROOT / 'Hate' / 'Hate_Others' / 'generated_images',
    'ambiguous':          PROJECT_ROOT / 'non-hate' / 'generated_images-ambigious',
    'counter_speech':     PROJECT_ROOT / 'non-hate' / 'generated_images-counter-speech',
    'neutral_discussion': PROJECT_ROOT / 'non-hate' / 'generated_images-neutral',
    'offensive_non_hate': PROJECT_ROOT / 'non-hate' / 'generated_images-offensive-non-hate',
}

# Map class_label → directory, then resolve {original_sample_id}.png
```

#### Step 2: Image Transforms (Train)
```python
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.45782750, 0.40821073],  # CLIP ViT-B/32 stats
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])
```

#### Step 3: Image Transforms (Eval)
```python
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.45782750, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
])
```

#### Step 4: Dataset & DataLoader
```python
class HateSpeechImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = (self.df['polarity'] == 'hate').astype(int).values  # Binary
        self.groups = self.df['target_group'].map(GROUP2ID).fillna(N_GROUPS-1).astype(int).values
        
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, group  # Multi-output for adversarial head

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
```

### 5.3 Cross-Modal Fusion Preprocessing

#### Step 1: Extract Text Embeddings
```python
# Load pre-trained MiniLM (frozen)
text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
text_embeddings = text_encoder.encode(texts, show_progress_bar=False)  # (N, 384)

# Cache for reuse (avoid recomputation)
joblib.dump(text_embeddings, cache_path)
```

#### Step 2: Extract Image Features (CLIP ViT-B/32)
```python
from clip import load as load_clip

model, preprocess = load_clip("ViT-B/32", device="cuda")
model.eval()

with torch.no_grad():
    image_features = model.encode_image(image_tensor)  # (N, 512)
    # Project to 384-dim for alignment with text
    proj = nn.Linear(512, 384)
    image_384 = proj(image_features)
```

#### Step 3: Concatenate for Fusion Classifiers
```python
# Late fusion: separate predictions from text & image, then combine
combined_features = np.concatenate([text_embeddings, image_384], axis=1)  # (N, 768)

# Stacking fusion: train meta-learner on OOF predictions
# Learned fusion: search for weight w that maximizes F1_val
```

---

## 6. EVALUATION & STATISTICAL TESTING SCRIPTS

### 6.1 Enhanced Statistical Tests
**File:** [`analysis/enhanced_statistical_tests.py`](analysis/enhanced_statistical_tests.py)

#### Test Suite
1. **Chi-Squared Test of Proportions (PRIMARY)**
   - Null hypothesis: FPR equal across identity groups
   - Test statistic: χ² = Σ (observed - expected)² / expected
   - Result: χ²=77.81, p=3.84×10⁻¹⁴ for text CF (highly significant)

2. **Kruskal-Wallis H-Test (SECONDARY ROBUSTNESS)**
   - Non-parametric alternative to ANOVA
   - Test statistic: H = 77.64
   - p-value: 4.17×10⁻¹⁴

3. **OLS Linear Probability Model**
   - Specification: FP ~ C(group) + C(condition) + C(group):C(condition)
   - F-statistic: 9.817
   - p-value: 1.72×10⁻¹⁰

4. **Per-Group FPR with Clopper-Pearson Exact Binomial CI**
   - 95% confidence intervals on FP indicators per group
   - Accounts for small sample sizes via exact method
   - Prevents normal approximation errors

5. **Pairwise Mann-Whitney U Tests**
   - Holm-Bonferroni correction (α/k)
   - Benjamini-Hochberg FDR correction
   - Multiple comparison penalty applied

6. **Fisher's Exact Test** (small groups <30)
   - Two-sided test for independence
   - No distributional assumptions

7. **Cohen's d Effect Size**
   - d = (μ₁ - μ₂) / pooled_std
   - Interpretation: small (0.2), medium (0.5), large (0.8)

8. **Logistic Regression per Modality**
   - Binary outcome: FP indicator
   - Covariates: C(group), C(condition)
   - Interaction term: C(group) × C(condition)

### 6.2 Image Leakage Audit
**File:** [`analysis/image_leakage_audit.py`](analysis/image_leakage_audit.py)

**Tests:**
1. Train/test split disjointness (ID level)
2. Exact image reuse detection (file hash comparison)
3. Variant leakage (original_sample_id level)
4. Per-group image count balance
5. Shortcut artifact detection (spurious correlations)

### 6.3 Cross-Modal Calibration Analysis
**File:** [`analysis/calibration_analysis.py`](analysis/calibration_analysis.py)

**Calibration Methods:**
1. Temperature Scaling: p_scaled = 1 / (1 + exp(-(logit / T)))
2. Isotonic Regression: Monotonic CDF fitting
3. Platt Scaling: Logistic regression on scores

**Metrics:**
- Expected Calibration Error (ECE): √(Σ |acc_bin - conf_bin|² · n_bin / N)
- Maximum Calibration Error (MCE): max |acc_bin - conf_bin|
- Brier Score: MSE of probability estimates
- Negative Log-Likelihood

### 6.4 MLP Cross-Validation (Text Models)
**File:** [`analysis/mlp_cross_validation.py`](analysis/mlp_cross_validation.py)

**Setup:**
- Text encoder: MiniLM-L12-v2 (frozen, 384-dim)
- Classifier: 2-layer MLP (384 → 128 → 2)
- Cross-validation: 5-fold stratified
- Validation set: Used for threshold tuning
- Test set: Held-out, unified 900-sample set

**Bootstrap Confidence Intervals:**
```python
n_bootstrap = 1500
bootstrap_samples = np.random.choice(y_true, size=(n_bootstrap, len(y_true)), replace=True)
# Calculate F1, AUC, FPR for each sample
# Extract 95% CI: [2.5%, 97.5%] percentiles
```

### 6.5 Per-Group Fairness Analysis
**File:** [`analysis/per_group_text_dfpr.py`](analysis/per_group_text_dfpr.py)

**Target Groups (8):**
1. race/ethnicity
2. religion
3. gender
4. sexual_orientation
5. national_origin/citizenship
6. disability
7. age
8. multiple/none

**Metrics per Group:**
- FPR (False Positive Rate on non-hate samples): FP / (FP + TN)
- FNR (False Negative Rate on hate samples): FN / (FN + TP)
- Count of samples in each group
- Equalized Odds Difference: max(|FPR_g1 - FPR_g2|) across all pairs
- Demographic Parity Difference: |P(pred=1 | g1) - P(pred=1 | g2)|

---

## 7. CONFIGURATION FILES & HYPERPARAMETERS

### 7.1 Image Model Configuration
**File:** [`image_models/config.py`](image_models/config.py)
```python
ADV_WEIGHT = 0.5  # Gradient Reversal Layer weight for adversarial loss
ADV_WEIGHT_INFORMAL_RANGE = (0.3, 0.7)  # Informally tested, no systematic sweep
```

### 7.2 Image Training Hyperparameters
**File:** [`image_models/train.py`](image_models/train.py)
```python
DEFAULT_CONFIG = {
    'epochs':          20,
    'batch_size':      64,
    'lr_backbone':     1e-4,       # Lower LR for frozen/pre-trained backbone
    'lr_heads':        1e-3,       # Higher LR for newly initialized heads
    'weight_decay':    1e-4,       # L2 regularization
    'label_smoothing': 0.05,       # Soft targets: (1-ε) · y + ε · (1-y)
    'adv_weight':      0.5,        # GRL loss scaling (in CF+GRL condition)
    'patience':        5,          # Early stopping patience
    'min_delta':       1e-4,       # Minimum improvement threshold
    'freeze_blocks':   0,          # Number of initial blocks to freeze (0=none)
    'architecture':    'clip_vit_b32',
    'freeze_encoder':  True,       # Freeze CLIP backbone
    'dropout':         0.3,        # Dropout on classifier head
    'num_workers':     2,          # DataLoader workers
    'grad_clip':       1.0,        # Gradient clipping max norm
}

# Learning rate schedule: CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# Early stopping criterion: F1 score on validation set
early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
```

### 7.3 Text Model Hyperparameters
**File:** [`text_models/train_models.py`](text_models/train_models.py)
```python
# TF-IDF Configuration
TFIDF_CONFIG = {
    'max_features': 10_000,
    'ngram_range': (1, 2),
    'lowercase': True,
    'stop_words': 'english',
    'min_df': 2,
    'max_df': 0.95,
    'sublinear_tf': True,
}

# Classifier Hyperparameters
# Logistic Regression
C = 1.0
max_iter = 1000
solver = 'lbfgs'

# Support Vector Machine
kernel = 'rbf'
gamma = 'scale'
C = 1.0

# Random Forest
n_estimators = 100
max_depth = 10
min_samples_split = 5
random_state = 42

# HateBERT Fine-tuning
HATEBERT_CONFIG = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 32,
    'weight_decay': 0.01,
    'max_length': 128,
    'warmup_steps': 500,
    'metric_for_best_model': 'macro_f1',
}
```

### 7.4 Baseline Pipeline Configuration
**File:** [`baseline-pipeline/config.py`](baseline-pipeline/config.py)
```python
# Data splits (different from canonical, for baseline only)
SPLIT_RATIOS = (0.60, 0.15, 0.25)  # train / val / test
SEEDS = [42, 123, 456]              # Multi-seed robustness

# Text Pipeline
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_C = 1.0
RF_N_ESTIMATORS = 100

# Image Pipeline
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Standard ImageNet stats
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 16
MAX_EPOCHS = 7
EARLY_STOP_PATIENCE = 3
```

### 7.5 Counterfactual Generation Configuration
**File:** [`src/counterfactual_gen/config.py`](src/counterfactual_gen/config.py)
```python
# Dataset Parameters
DATASET_NAME = "ucberkeley-dlab/measuring-hate-speech"
TOTAL_SAMPLES = 6000
SAMPLES_PER_CLASS = 750

# Hate Classes (binary positive)
HATE_CLASSES = {
    'hate_race', 'hate_religion', 'hate_gender', 'hate_other'
}

# Non-Hate Classes (binary negative)
NON_HATE_CLASSES = {
    'offensive_non_hate', 'neutral_discussion', 'counter_speech', 'ambiguous'
}

# Quality Control
MIN_TEXT_LENGTH = 10  # words
MAX_TEXT_LENGTH = 200
HATE_SCORE_THRESHOLD_HIGH = 0.6
MIN_CONFIDENCE = 3  # Annotators agreeing
DUPLICATE_SIMILARITY_THRESHOLD = 0.95

RANDOM_SEED = 42
```

---

## 8. CANONICAL SPLITS IMPLEMENTATION

### 8.1 Split Strategy
**File:** [`canonical_splits.py`](canonical_splits.py)

**Purpose:** Single source of truth for all train/val/test splits across pipelines

**Strategy:**
1. Extract originals only from full dataset
2. Remove non-English samples (>5% non-ASCII)
3. Stratify on `class_label` (8-class)
4. Split at group level: each `original_sample_id` → exactly one split
5. Target ratio: 70% train / 15% val / 15% test
6. Persistent cache: `data/splits/canonical_splits.json`

### 8.2 Implementation
```python
def _build_splits(df_path: str | None = None) -> dict[str, Any]:
    df = pd.read_csv(df_path)
    originals = df[df['cf_type'] == 'original'].copy()
    
    # 70/15/15 stratified split
    train_orig, temp_orig = train_test_split(
        originals,
        test_size=0.30,
        stratify=originals["class_label"],
        random_state=42,
    )
    val_orig, test_orig = train_test_split(
        temp_orig,
        test_size=0.50,
        stratify=temp_orig["class_label"],
        random_state=42,
    )
    
    # Return ID sets (not dataframes)
    return {
        "train_ids": set(train_orig["original_sample_id"]),
        "val_ids": set(val_orig["original_sample_id"]),
        "test_ids": set(test_orig["original_sample_id"]),
    }

def build_condition_split_frames(
    df: pd.DataFrame,
    condition: str,  # "ncf" or "cf"
    splits: dict[str, set[str]] | None = None,
    augment_val_for_cf: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Build condition-specific train/val/test frames.
    
    For nCF: All splits contain originals only
    For CF:  Train includes originals + counterfactuals
             Val/test contain originals only (for fair evaluation)
    """
    if splits is None:
        splits = get_canonical_splits()
    
    train_df = df[df['original_sample_id'].isin(splits['train_ids'])]
    val_df = df[df['original_sample_id'].isin(splits['val_ids'])]
    test_df = df[df['original_sample_id'].isin(splits['test_ids'])]
    
    # For nCF condition, keep only originals in train too
    if condition == "ncf":
        train_df = train_df[train_df['cf_type'] == 'original'].copy()
    
    # Val/test: always originals only
    val_df = val_df[val_df['cf_type'] == 'original'].copy()
    test_df = test_df[test_df['cf_type'] == 'original'].copy()
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
    }
```

### 8.3 Realized Split Statistics
```
Metadata from canonical_splits.json:
  random_state: 42
  n_train: 4,158 originals (69.99%)
  n_val: 891 originals (15.00%)
  n_test: 892 originals (15.01%)
  stratify_on: class_label
  
Class Distribution (train):
  hate_race: 525
  hate_religion: 522
  hate_gender: 518
  hate_other: 523
  offensive_non_hate: 525
  neutral_discussion: 522
  counter_speech: 525
  ambiguous: 498
  
Split Integrity Asserts:
  ✓ train_ids ∩ val_ids = ∅
  ✓ train_ids ∩ test_ids = ∅
  ✓ val_ids ∩ test_ids = ∅
```

---

## 9. KEY RESULT FILES & OUTPUTS

### 9.1 Dataset Files
| File | Rows | Purpose |
|------|------|---------|
| `data/datasets/final_dataset_18k.csv` | 18,000 | Master dataset (originals + counterfactuals) |
| `src/counterfactual_gen/hate_speech_dataset_6k.csv` | 6,000 | Original samples only |
| `data/datasets/final_dataset_18k_t2i_prompts.csv` | 18,000 | Alternative naming (same as 18k.csv) |
| `data/splits/canonical_splits.json` | Metadata | Train/val/test ID assignments |

### 9.2 Text Model Results
| File | Contents | Generator |
|------|----------|-----------|
| `text_models/enhanced_results/enhanced_results.json` | TF-IDF + MiniLM results | `enhanced_analysis.py` |
| `text_models/binary_fairness_results/binary_fairness_results.json` | Binary TF-IDF results | `binary_fairness_analysis.py` |
| `text_models/enhanced_results/models/minilm_mlp_*.joblib` | Trained MLP classifiers | `train_models.py` |

### 9.3 Image Model Results
| File | Contents | Generator |
|------|----------|-----------|
| `image_models/results/evaluation_results.json` | CLIP/EfficientNet results | `evaluate.py` |
| `image_models/models/*.pth` | Trained checkpoints | `train.py` |

### 9.4 Cross-Modal Fusion Results
| File | Contents | Generator |
|------|----------|-----------|
| `cross_modal/results/late_fusion_results.json` | Late fusion metrics | `late_fusion_ensemble.py` |
| `cross_modal/results/stacking_ensemble_results.json` | Stacking meta-learner | `stacking_ensemble.py` |
| `cross_modal/results/cross_attention_fusion_results.json` | CrossAttention+GRL model | `cross_attention_fusion.py` |

### 9.5 Analysis Results
| File | Contents | Generator |
|------|----------|-----------|
| `analysis/results/enhanced_statistical_tests.json` | Chi-squared, ANOVA, Cohen's d | `enhanced_statistical_tests.py` |
| `analysis/results/mlp_cv_results.json` | MiniLM bootstrap CI | `mlp_cross_validation.py` |
| `analysis/results/multi_seed_results.json` | EfficientNet multi-seed | `scripts/multi_seed_experiment.py` |

---

## 10. SUMMARY & KEY TAKEAWAYS

### Major Findings
1. **CAD fairness benefit is architecture-conditional:**
   - HateBERT E2E fine-tuning: FPR improves 20.3% → 17.8% with CAD
   - Frozen encoders: Limited benefit; may require more data
   - Image models: Require explicit GRL debiasing to prevent fairness regression

2. **Image generation quality matters:**
   - T2I biases can leak into classifiers (polarity-conditional lighting)
   - Fix reduced F1 gap from 3.2% artificial → 3.5% realistic

3. **Train/test split discipline is critical:**
   - Original variant leakage inflated fairness metrics by ~8% EO-diff
   - Canonical splits prevent this via group-level assignment

4. **Multimodal fusion effective:**
   - Late fusion (w=0.5): F1=0.853, AUC=0.910, ECE=0.052 (post-calibration)
   - Cross-attention+GRL: Similar performance, learned gating

### Reproducibility Notes
- All splits fixed via `canonical_splits.py` (seed=42)
- Data leakage issues documented in `APPENDIX_EXTENDED.md`
- Code locations traced throughout this report
- Configuration files centralized in config.py modules

### Contact & Credits
- Project: ACM Multimedia 2026 submission
- Primary scripts: Text (`enhanced_analysis.py`), Images (`run_all.py`), Fusion (`late_fusion_ensemble.py`)
- Analysis: `analysis/run_all.py` aggregates all statistical tests

---

**Report Generated:** April 2026  
**Last Updated:** See git history or paper_sections.md
