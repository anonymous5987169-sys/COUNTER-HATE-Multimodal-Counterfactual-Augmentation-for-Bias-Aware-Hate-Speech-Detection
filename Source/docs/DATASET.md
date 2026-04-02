# DATASET: MIDAS Construction and Specification

This document details the construction, composition, and specification of MIDAS (Multimodal Identity-Driven Augmentation Study), the 18,000 image-text pair dataset at the core of this work.

---

## Overview

MIDAS combines;

- **6,000 original text samples** from Kennedy et al. (UC Berkeley D-Lab; measuring-hate-speech repository)
- **12,000 counterfactually rewritten text samples** (2 per original) via Qwen2.5-7B-Instruct
- **18,000 synthetically generated images** (one per text sample) via Z-Image-Turbo

All examples are stratified across 8 protected identity group categories at exactly 750 samples per class in the base corpus.

---

## Base Corpus (Kennedy et al., 6,000 Originals)

### Source

The base corpus is drawn from Kennedy et al. [12], sourced from the UC Berkeley D-Lab hate content collection (measuring-hate-content repository). The 6,000-sample corpus is stratified across eight protected-group classes at exactly 750 samples each.

### Class Distribution and Binary Mapping

| Class | Identity Category | Binary Label | Count |
|---|---|---|---|
| hate_race | Racial; ethnic groups | 1 (hate) | 750 |
| hate_religion | Religious groups | 1 (hate) | 750 |
| hate_gender | Gender; sexual orientation | 1 (hate) | 750 |
| hate_other | Other protected groups | 1 (hate) | 750 |
| offensive_non_hate | Offensive non-targeted content | 0 (non-hate) | 750 |
| counter_content | Content countering hate | 0 (non-hate) | 750 |
| neutral_discussion | Neutral factual discussion | 0 (non-hate) | 750 |
| ambiguous | Ambiguous; borderline | 0 (non-hate) | 750 |
| **Total** | — | — | **6,000** |

Hate-content labels are binary annotations derived from hatescore thresholds and target-group metadata provided in the original corpus. Duplicate filtering and confidence-aware balancing are applied prior to class selection. Each sample carries fields; sample_id, class_label, target_group, and polarity.

**Critically;** The polarity field is withheld from the LLM rewriting prompt to prevent label-correlated lexical leakage (see Counterfactual Generation Pipeline).

---

## Counterfactual Text Generation Pipeline

### Overview

Counterfactual text generation proceeds via **Qwen2.5-7B-Instruct** over the full 6,000-sample corpus on Kaggle (T4×2 GPU). Each original sample yields exactly **two counterfactual variants**, tripling the corpus to 18,000 text samples.

### Algorithm (Simplified)

For each sample (x_i, y_i, g_i) in the original corpus;

1. **Detect identity terms** in x_i via regex dictionaries (race, religion, gender, sexuality, national origin, disability, age)

2. **Select prompt mode**;
   - **Explicit mode** (identity terms detected); Build swap prompt; substitute target group term from same category
   - **Implicit mode** (contextual identity); Build rewrite prompt; request full paraphrase with shifted demographic reading

3. **Generate two counterfactuals** x^(1)_i, x^(2)_i via Qwen2.5-7B-Instruct;
   - temperature=0.25, top-p=0.9, max_new_tokens=128, repetition_penalty=1.1

4. **Post-processing and validation**;
   - Strip &lt;think&gt; tokens and LLM preamble artifacts
   - Apply CJK character guard; discard outputs containing non-ASCII characters outside expected token set
   - Validate output; check length ratio, label-preserving structure, absence of polarity-correlated descriptors
   - **Fallback**; if validation fails, apply deterministic regex-based term substitution

5. **Output**; Record triplet {(x^(0)_i, y_i, g_i), (x^(1)_i, y_i, g^(1)_i), (x^(2)_i, y_i, g^(2)_i)} with shared original_sample_id = i

### Engineering Challenges During Generation

**CUDA Multiprocessing Conflicts.** vLLM inference server initialization inside a forked process caused deadlocks on Kaggle. **Fix;** Restructure pipeline to initialize engine in main process before launching data-parallel workers.

**Out-of-Memory (OOM) Failures.** Batched generation caused VRAM exhaustion. **Fix;** Recursive batch-halving; reduce batch size by half on each OOM signal until success or minimum size of one.

**CJK Character Injection.** Fraction of LLM outputs contained CJK (Chinese, Japanese, Korean) characters and non-ASCII refusal tokens, triggered by demographic terms matching non-English training-data patterns. **Fix;** Character-set guard post-generation; discard any output with non-Latin non-punctuation Unicode characters outside expected token set; replace with deterministic fallback.

**Test-Set Contamination.** Early split logic assigned counterfactuals independently of source original, producing test-set leakage where variants appeared in both train and test. **Fix;** Revise canonical split logic to key assignment on original_sample_id, ensuring all three variants remain in same partition.

---

## Identity Mapping and Swap Pairs

Eight protected-group categories are defined; swap pairs are drawn from same category to preserve label structure.

| Category | Representative Terms | Example Swap Pair |
|---|---|---|
| Race; Ethnicity | Black, White, Asian, Hispanic | Black ↔ Asian |
| Religion | Muslim, Christian, Jewish, Hindu | Muslim ↔ Christian |
| Gender | women, men, girls, boys | women ↔ men |
| Sexual Orientation | gay, straight, LGBTQ+, lesbian | gay ↔ straight |
| National Origin | immigrant, refugee, Mexican | immigrant ↔ citizen |
| Disability | disabled, autistic, blind | disabled ↔ able-bodied |
| Age | elderly, young, teen | elderly ↔ young |
| Multiple; None | compound identity or absent | context-driven rewrite |

**Cross-category swaps are not generated** because substituting a racial term with a gendered term changes the target group in ways that confound fairness analysis. **Intersectional identities are not exhaustively covered;** this is acknowledged as a limitation.

---

## Text-to-Image Generation

### Overview

Each of the 18,000 text samples is passed as an image prompt to **Z-Image-Turbo**, producing a 720×720 RGB image. Generation ran on Lightning AI H200 (141 GB VRAM) using a ComfyUI-based workflow with torch.compile and batch prompt encoding at approximately 7.7 prompts/sec.

### Generation Parameters

| Parameter | Value |
|---|---|
| Model | Z-Image-Turbo (FP8 diffusion; Qwen3-4B CLIP backbone) |
| Resolution | 720 × 720 px |
| Diffusion steps | 9 (Euler sampler) |
| CFG scale | 1.0 |
| Schedule | Simple |
| Denoise strength | 1.0 |
| Batch size | 256 (adaptive OOM halving) |
| Deterministic seed | (SEED_BASE + i × SEED_PRIME) mod 2^32 |
| Output format | PNG; stored by binary label directory |

### Prompt Engineering

Prior to passing texts directly to the diffusion model, prompt enhancement converts each text row into a structured visual description specifying resolution, camera angle, lighting, and scene composition.

**Prompt structure;**

1. **Fixed quality prefix;** "8k uhd, ultra high resolution, photorealistic, RAW photo, masterpiece, best quality, detailed, professional, sharp focus, cinematic composition, depth of field"

2. **Row-level scene description;** Converted from text sample; e.g., "A scene depicting conversations between people, professional lighting, depth of field, contemporary setting"

3. **Strict anti-text negative block;** "text, written text, letters, words, numbers, typography, font, readable text, handwriting, graffiti, holding sign, holding poster, holding board, holding banner, holding placard, label, sticker, badge, caption, subtitle, watermark, logo, content bubble, thought bubble"

4. **General quality negative suffix;** "low quality, worst quality, blurry, deformed, distorted, bad anatomy, cartoon, anime, oversaturated, jpeg artifacts"

### Visual Tone Bias Check

A visual tone bias check was performed on a 200-sample audit subset before generating the full corpus. Early prompt templates included the word "violent" in hate-class prompts and "peaceful" in non-hate-class prompts, introducing systematic lighting and environmental differences. **Fix;** Revise templates to neutral scene descriptors; remove polarity-indicative adjectives from all prompt fields. Final audit confirmed no statistically significant difference in mean image brightness or colour saturation between hate and non-hate classes.

### Checkpoint and Resumption

Checkpoint records are flushed to CSV every five batches, enabling resumption from interruption without re-generating completed images. Deterministic seed formula ensures each image is reproducible from its row index. Batch-level OOM failures trigger recursive batch-halving; per-image VAE decode fallback is applied when batch-level decoding fails but individual decoding succeeds.

---

## Data Splitting Strategy

### Canonical Splits

Canonical stratified splits are produced at the original_sample_id level with random seed 42. Stratification is on class_label (binary), ensuring approximately 50% hate samples in each partition.

| Partition | Original IDs | Total Rows (CF inc.) | Ratio |
|---|---|---|---|
| Train | 4,158 | 12,474 | 69.99% |
| Val | 891 | 891 | 15.00% |
| Test | 892 | 892 | 15.01% |
| **Total** | **5,941** | **14,257** | **100%** |

### Key Properties

- Valid and test partitions contain **originals only**; counterfactual variants are confined to the training partition of their source original
- Counterfactual variants of a given original_sample_id always appear in the training fold of that original
- Test set is held fixed across all experiments and is never modified
- API function; `from canonical_splits import get_canonical_splits()` provides single source of truth

---

## Dataset Quality and Limitations

### Strengths

- **Balanced;** 50% hate, 50% non-hate in each partition
- **Identity coverage;** 8 categories; 750 samples per category in base corpus
- **Deterministic;** Reproducible splits (seed=42), deterministic image seeds
- **Controlled;** Polarity field withheld; label-correlated lexical leakage prevented
- **Validated;** CJK guard; anti-text negatives in image generation; tone bias audit

### Limitations

- **Synthetic images;** T2I images synthesized from tweet text; do not faithfully represent real-world social media image content
- **Small test set;** 892 samples limits statistical power for rarer identity groups (though fixed and stratified, appropriate for leaderboard fairness)
- **LLM bias;** Counterfactual generation relies on LLM identity-term rewrites; intersectional identities and context-dependent slurs not handled
- **Diffusion bias;** Z-Image-Turbo's own demographic biases propagate into images; GRL training only partially counteracts

---

## Dataset Availability

MIDAS will be released on HuggingFace Hub as `vs16/counter-hate-dataset` upon acceptance. Training and validation sets are public; test set split is fixed and provided to enable reproducible leaderboard evaluation.

---

## References

Kennedy, B., et al. (2020). Measuring the Reliability of Hate Speech Annotations; The Case of the European Parliament Debate. Proceedings of the 2020 Conference on Computational Linguistics, 6313–6325.
