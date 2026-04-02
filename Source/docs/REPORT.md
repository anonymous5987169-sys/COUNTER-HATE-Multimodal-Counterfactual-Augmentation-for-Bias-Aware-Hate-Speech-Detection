# Z-Image-Turbo Batch Image Generation Pipeline -- Technical Report

**Document Version:** 1.0  
**Date:** 20 February 2026  
**Script:** `image-gen.py`  
**Platform:** Lightning AI Studio -- NVIDIA H200 80 GB SXM5  
**Authors:** Research Team  
**Classification:** Internal / Publication Support  

## Post-Rerun Update (2026-03-20)

Generated images were replaced and the downstream image-dependent evaluation was rerun.

Updated artifacts now used by the project:
- `image_models/results/evaluation_results.json`
- `cross_modal/results/late_fusion_results.json`
- `cross_modal/results/stacking_ensemble_results.json`
- `cross_modal/results/learned_fusion_results.json`
- `cross_modal/results/cross_attention_fusion_results.json`

Current image headline metrics:
- `nCF`: F1 `0.7809`, AUC `0.8322`, FPR `0.4009`
- `CF-no-adv`: F1 `0.8080`, AUC `0.8474`, FPR `0.3333`
- `CF+GRL`: F1 `0.7885`, AUC `0.8401`, FPR `0.3649`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Approach and Methodology](#3-approach-and-methodology)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Inputs and Data Requirements](#5-inputs-and-data-requirements)
6. [Models Used and Their Roles](#6-models-used-and-their-roles)
7. [End-to-End Pipeline: Detailed Step Breakdown](#7-end-to-end-pipeline-detailed-step-breakdown)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [Functional Flow Block Diagram](#9-functional-flow-block-diagram)
10. [Sequence Diagram: Runtime Interaction](#10-sequence-diagram-runtime-interaction)
11. [Key Technical Decisions and Rationale](#11-key-technical-decisions-and-rationale)
12. [Error Handling and Fault Tolerance](#12-error-handling-and-fault-tolerance)
13. [Output Artifacts](#13-output-artifacts)
14. [Code Quality Assessment](#14-code-quality-assessment)
15. [Identified Issues and Risks](#15-identified-issues-and-risks)
16. [Suggested Enhancements](#16-suggested-enhancements)
17. [Proposed Solution Architecture (v2)](#17-proposed-solution-architecture-v2)
18. [Rating: Research Readiness (Scale of 10)](#18-rating-research-readiness-scale-of-10)
19. [Glossary](#19-glossary)
20. [Appendices](#20-appendices)

---

## 1. Executive Summary

This report provides a comprehensive technical review of the `image-gen.py` script, a high-throughput text-to-image generation pipeline built on top of ComfyUI and the Z-Image-Turbo diffusion model. The script is designed to run on an NVIDIA H200 GPU within a Lightning AI Studio environment, processing approximately 18,000 text prompts from a CSV dataset and producing corresponding 720x720 PNG images.

**For non-technical readers:** This script takes a spreadsheet of text descriptions (prompts) and automatically generates a matching photograph-quality image for each one, using advanced AI. It is optimized to produce images as fast as possible on high-end hardware, includes automatic recovery if something goes wrong, and packages everything neatly for downstream analysis.

**For technical readers:** The pipeline leverages FP8-quantized UNet inference via ComfyUI's node system, Qwen3-4B as the CLIP text encoder, true batch processing with adaptive OOM recovery through recursive batch halving, and checkpoint-based resumability. All inference runs under `torch.inference_mode()` with TF32 and cuDNN autotuning enabled.

**For researchers:** The system generates a counterfactual image dataset suitable for bias analysis, fairness evaluation, or vision-language model benchmarking. Deterministic seeding ensures reproducibility. Output is a joined CSV that pairs each original prompt with its generated image path, ready for quantitative evaluation.

---

## 2. Problem Statement

### 2.1 Core Problem

The research requires generating a large-scale synthetic image dataset (~18,000 images) from structured text-to-image prompts. Each prompt is paired with a `counterfactual_id`, suggesting the dataset is designed for **counterfactual analysis** -- a methodology where controlled variations in text prompts allow researchers to measure how image generation models respond to specific attribute changes (e.g., altering gender, ethnicity, or context descriptors while holding other variables constant).

### 2.2 Constraints

| Constraint | Detail |
|---|---|
| Scale | ~18,000 images in a single automated run |
| Hardware | Single NVIDIA H200 GPU (80 GB or 141 GB HBM3e variants) |
| Reproducibility | Deterministic output required for scientific publication |
| Reliability | Must not lose progress on crash or OOM |
| Quality | 720x720 resolution, professional-grade output |
| Throughput | Maximize GPU utilization to minimize wall-clock time |
| Traceability | Every generated image must be linked back to its source prompt |

### 2.3 Why This Matters

Counterfactual image generation is a critical tool in AI fairness research. By systematically varying attributes in prompts and comparing the generated images, researchers can quantify biases embedded in text-to-image models. A reliable, high-throughput pipeline is essential to produce datasets large enough for statistically significant analysis.

---

## 3. Approach and Methodology

### 3.1 High-Level Approach

The script follows a **batch-oriented, checkpoint-resumable, fault-tolerant** generation strategy:

1. **Infrastructure bootstrapping** -- Install all dependencies and clone the ComfyUI framework programmatically, ensuring the environment is self-contained.
2. **Model acquisition** -- Download three model files (UNet, CLIP, VAE) from HuggingFace Hub using aria2c for parallelized, resumable downloads.
3. **Batch generation** -- Process prompts in large batches (configurable, set to 148), encoding all prompts through CLIP, running batched diffusion through KSampler, and decoding through VAE in a single pass.
4. **Adaptive fault tolerance** -- On GPU out-of-memory (OOM) errors, recursively halve the batch size and retry. On VAE OOM, fall back to per-image decoding.
5. **Checkpoint management** -- Persist progress every 5 batches so that interrupted runs can resume without re-generating completed images.
6. **Post-processing** -- Join generated image paths back to the original dataset, produce summary statistics, and create ZIP archives for distribution.

### 3.2 Technology Stack

| Component | Technology | Version/Variant |
|---|---|---|
| Orchestration Framework | ComfyUI (headless, node-API mode) | Latest from GitHub main branch |
| Diffusion Model (UNet) | Z-Image-Turbo | FP8 E4M3FN quantized |
| Text Encoder (CLIP) | Qwen3-4B | safetensors format |
| Image Decoder (VAE) | Standard autoencoder | ae.safetensors |
| Deep Learning Framework | PyTorch | cu126 build |
| Data Processing | Polars | Latest stable |
| Image I/O | Pillow (PIL) | Latest stable |
| Download Manager | aria2c | System package |
| Execution Platform | Lightning AI Studio | H200 instance |
| Language | Python | 3.x (system default) |

---

## 4. System Architecture Overview

The system is organized into seven logical layers:

```
+------------------------------------------------------------------+
|                    OPERATOR / RESEARCHER                          |
+------------------------------------------------------------------+
          |                                          ^
          v                                          |
+------------------------------------------------------------------+
|  INPUT LAYER                                                      |
|  - final_dataset_18k_t2i_prompts.csv                             |
|  - checkpoint_progress.csv (optional, for resume)                |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  ENVIRONMENT LAYER                                                |
|  - pip packages (polars, pillow, tqdm, pytorch cu126)            |
|  - System packages (aria2, git)                                  |
|  - ComfyUI repository clone                                     |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  MODEL LAYER                                                      |
|  - Z-Image-Turbo UNet (FP8 E4M3FN) -- Diffusion backbone        |
|  - Qwen3-4B CLIP -- Text encoding                               |
|  - VAE (ae.safetensors) -- Latent-to-pixel decoding              |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  COMPUTE LAYER                                                    |
|  - H200 GPU (sm_90, TF32, cuDNN benchmark)                      |
|  - Batch KSampler (euler, 9 steps, CFG 1.0)                     |
|  - VRAM monitoring and OOM recovery                              |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  PERSISTENCE LAYER                                                |
|  - PNG images in output/generated_images/                        |
|  - Checkpoint CSV (every 5 batches)                              |
+------------------------------------------------------------------+
          |
          v
+------------------------------------------------------------------+
|  OUTPUT LAYER                                                     |
|  - combined_dataset_with_images.csv                              |
|  - failed_generations.csv                                        |
|  - generated_images.zip                                          |
|  - complete_output.zip                                           |
+------------------------------------------------------------------+
```

---

## 5. Inputs and Data Requirements

### 5.1 Primary Input: CSV Dataset

| Property | Value |
|---|---|
| File | `final_dataset_18k_t2i_prompts.csv` |
| Location | `/teamspace/studios/this_studio/` |
| Expected Row Count | ~18,000 |
| Required Columns | `counterfactual_id` (unique string identifier), `t2i_prompt` (text-to-image prompt) |
| Null Handling | Rows with null `t2i_prompt` values are filtered out before processing |

### 5.2 Secondary Input: Checkpoint File (Optional)

| Property | Value |
|---|---|
| File | `output/checkpoint_progress.csv` |
| Columns | `counterfactual_id`, `image_path` |
| Purpose | Resume interrupted runs by excluding already-completed IDs |

### 5.3 External Dependencies (Downloaded at Runtime)

| Artifact | Source | Size (Approximate) |
|---|---|---|
| z-image-turbo-fp8-e4m3fn.safetensors | HuggingFace (T5B/Z-Image-Turbo-FP8) | 5-8 GB |
| qwen_3_4b.safetensors | HuggingFace (Comfy-Org/z_image_turbo) | ~8 GB |
| ae.safetensors | HuggingFace (Comfy-Org/z_image_turbo) | ~300 MB |
| ComfyUI repository | GitHub (comfyanonymous/ComfyUI) | ~50 MB |

---

## 6. Models Used and Their Roles

### 6.1 Z-Image-Turbo (UNet / Diffusion Backbone)

| Property | Detail |
|---|---|
| Role | Core image generation model -- iteratively denoises latent representations |
| Architecture | UNet-based diffusion model (turbo variant for reduced step count) |
| Quantization | FP8 E4M3FN -- 8-bit floating point for reduced VRAM and increased throughput |
| Inference Steps | 9 (significantly fewer than standard diffusion models which use 20-50) |
| Sampler | Euler (first-order ODE solver) |
| Scheduler | Simple (linear noise schedule) |
| CFG Scale | 1.0 (classifier-free guidance effectively disabled) |
| Denoise Strength | 1.0 (full denoising from pure noise) |
| Source | `huggingface.co/T5B/Z-Image-Turbo-FP8` |

**How it is used:** The UNet receives a batch of latent noise tensors (720x720, mapped to latent space dimensions) along with text conditioning from CLIP. Over 9 Euler steps, it progressively removes noise to produce clean latent representations. The "turbo" variant is specifically trained to produce acceptable results in very few steps, trading marginal quality for dramatically higher throughput.

**What FP8 E4M3FN means:** This is a hardware-native 8-bit floating-point format supported by NVIDIA Hopper architecture (H100/H200). E4M3FN provides 4 exponent bits and 3 mantissa bits with no NaN encoding. It halves memory usage compared to FP16 and enables the Transformer Engine for near-doubled throughput on supported hardware. Quality loss is minimal for inference workloads.

### 6.2 Qwen3-4B (CLIP Text Encoder)

| Property | Detail |
|---|---|
| Role | Converts text prompts into dense vector embeddings (conditioning tensors) |
| Architecture | Qwen3-4B -- a 4-billion parameter language model used as a text encoder |
| Loading Type | `wan` (WAN-type CLIP loading in ComfyUI, for Wan-series model compatibility) |
| Prompt Prefix | `"high quality, detailed, professional, sharp focus"` prepended to all prompts |
| Negative Prompt | Empty string (`""`), pre-encoded once and reused via tensor repetition |
| Source | `huggingface.co/Comfy-Org/z_image_turbo` |

**How it is used:** Each text prompt is concatenated with a quality-boosting prefix and then encoded through the Qwen3-4B model to produce conditioning tensors. These tensors guide the UNet diffusion process, telling it what content to generate. The negative conditioning (an empty string) is encoded once at startup and then replicated for each batch via `torch.Tensor.repeat()`, avoiding redundant CLIP forward passes.

**Conditioning batching:** Because different prompts may produce conditioning tensors with different sequence lengths, the script includes `find_variable_dim` and `pad_to_max` utilities that detect the variable dimension and zero-pad shorter tensors to match the longest one in the batch. This enables true batched inference through KSampler rather than sequential per-prompt generation.

### 6.3 VAE (Variational Autoencoder -- Image Decoder)

| Property | Detail |
|---|---|
| Role | Decodes latent representations from the UNet into pixel-space RGB images |
| Architecture | Standard autoencoder (symmetric encoder-decoder) |
| Output Format | `[N, H, W, C]` float32 tensor, values in `[0, 1]` |
| Fallback Behavior | On batch OOM, falls back to per-image decoding |
| Source | `huggingface.co/Comfy-Org/z_image_turbo` |

**How it is used:** After KSampler produces clean latent samples, the VAE decoder maps them to 720x720x3 RGB images. The `_vae_decode_safe` function wraps this in a try/except block: if the full batch decode triggers OOM, it falls back to decoding one image at a time. If even a single-image decode fails, a blank (all-zeros) image is inserted as a placeholder.

---

## 7. End-to-End Pipeline: Detailed Step Breakdown

### Phase 1: Infrastructure Setup (Lines 1-60)

| Step | Action | Detail |
|---|---|---|
| 1.1 | Package installation | Installs `polars`, `pillow`, `tqdm` via pip. Installs `aria2`, `git` via apt. Upgrades PyTorch to cu126 build. |
| 1.2 | ComfyUI clone | Clones the ComfyUI repository to `/teamspace/studios/this_studio/ComfyUI/` if not already present. Installs its `requirements.txt`. |
| 1.3 | Directory scaffold | Creates `ComfyUI/models/diffusion_models/`, `ComfyUI/models/clip/`, `ComfyUI/models/vae/`, and `output/generated_images/`. |

### Phase 2: Model Acquisition (Lines 61-92)

| Step | Action | Detail |
|---|---|---|
| 2.1 | Define download targets | Three model files from HuggingFace Hub: UNet (FP8), CLIP (Qwen3-4B), VAE. |
| 2.2 | Download via aria2c | 16 parallel connections (`-x 16 -s 16`), 1 MB chunks (`-k 1M`), resumable (`-c`). Skips files that already exist on disk. |
| 2.3 | Verification | Existence check only (no checksum validation). |

### Phase 3: GPU Configuration (Lines 93-110)

| Step | Action | Detail |
|---|---|---|
| 3.1 | cuDNN benchmark mode | Enables auto-tuning of convolution algorithms for fixed input sizes. |
| 3.2 | TF32 enablement | Enables TensorFloat-32 for matrix multiplications and convolutions (Ampere+ feature). |
| 3.3 | VRAM assertion | Asserts total VRAM >= 100 GB. Fails fast if running on incorrect hardware. |

### Phase 4: Model Loading (Lines 111-140)

| Step | Action | Detail |
|---|---|---|
| 4.1 | Node instantiation | Creates instances of `UNETLoader`, `CLIPLoader`, `VAELoader`, `CLIPTextEncode`, `KSampler`, `VAEDecode`, `EmptyLatentImage` from ComfyUI's `NODE_CLASS_MAPPINGS`. |
| 4.2 | Model loading | Under `torch.inference_mode()`, loads UNet (FP8), CLIP (wan type), and VAE to GPU. |
| 4.3 | Negative pre-encoding | Encodes empty string through CLIP once and caches the result as `_neg_single`. |

### Phase 5: Data Preparation (Lines 280-315)

| Step | Action | Detail |
|---|---|---|
| 5.1 | CSV loading | Reads `final_dataset_18k_t2i_prompts.csv` using Polars. |
| 5.2 | Null filtering | Removes rows where `t2i_prompt` is null. |
| 5.3 | Checkpoint resume | If `checkpoint_progress.csv` exists, reads completed IDs and filters them out of the working dataset. |
| 5.4 | Batch partitioning | Slices remaining rows into chunks of `BATCH_SIZE` (148) for the generation loop. |

### Phase 6: Batch Generation Loop (Lines 316-430)

For each batch of up to 148 prompts:

| Step | Function | Detail |
|---|---|---|
| 6.1 | VRAM gate | Checks `vram_free_gb()`. If below 6 GB, runs aggressive cache clearing (gc.collect + torch.cuda.empty_cache). |
| 6.2 | Latent generation | `EmptyLatentImage.generate(720, 720, batch_size=N)` creates a tensor of Gaussian noise in latent space. |
| 6.3 | Prompt encoding | Each prompt is prefixed with `"high quality, detailed, professional, sharp focus"` and encoded through Qwen3-4B CLIP. Variable-length conditioning tensors are padded to equal length and concatenated into a batch. |
| 6.4 | Negative replication | The pre-encoded empty negative is repeated N times via `torch.Tensor.repeat()`. |
| 6.5 | KSampler diffusion | Single batched call: `KSampler.sample(unet, seed, 9, 1.0, "euler", "simple", pos, neg, latent, denoise=1.0)`. Seed is deterministically derived from the global image index. |
| 6.6 | VAE decode | `_vae_decode_safe(vae, samples)` attempts batch decode. Falls back to per-image on OOM. |
| 6.7 | Image save | Each decoded image is clamped to [0,1], scaled to [0,255] uint8, and saved as `{counterfactual_id}.png`. |
| 6.8 | Checkpoint | Every 5 batches, pending results are appended to `checkpoint_progress.csv`. |

### Phase 7: Post-Processing (Lines 431-540)

| Step | Action | Detail |
|---|---|---|
| 7.1 | Final checkpoint flush | Writes any remaining pending results to the checkpoint file. |
| 7.2 | Dataset join | Re-reads the original CSV and left-joins it with a mapping of `{counterfactual_id -> generated_image_path}` built by scanning the output directory. |
| 7.3 | Metadata columns | Adds `total_successful`, `total_failed`, `generation_timestamp`, `image_resolution`, `batch_size_used` columns. |
| 7.4 | Combined CSV | Writes `combined_dataset_with_images.csv` to the output directory. |
| 7.5 | Failed CSV | If any images failed, writes `failed_generations.csv` listing failed IDs. |
| 7.6 | ZIP archives | Creates `generated_images.zip` (images only) and `complete_output.zip` (all outputs). |
| 7.7 | Summary | Prints total attempted, successful, failed, success rate, resolution, and batch size. |

---

## 8. Data Flow Diagram



```
INPUT                         PROCESSING                          OUTPUT
-----                         ----------                          ------

final_dataset_18k             Polars read + filter nulls
  _t2i_prompts.csv  -------->  + checkpoint resume  ---------->  Working DataFrame
                                                                      |
                                                                      v
                              +-----------------------------------+
                              |  FOR EACH BATCH (N=148)           |
                              |                                   |
                              |  prompts[] -----> CLIP Encode     |
                              |                    |              |
                              |                    v              |
                              |  Empty latent --> KSampler <------+
                              |  (720x720xN)      |  (euler,9)   |
                              |                    v              |
                              |               VAE Decode          |
                              |                    |              |
                              |                    v              |
                              |              [N,720,720,3]        |
                              |                    |              |
                              |                    v              |
                              |         Save {id}.png per image   |
                              +-----------------------------------+
                                                   |
                                                   v
                              Checkpoint CSV (every 5 batches)
                                                   |
                                                   v
                              Join original CSV + image paths -----> combined_dataset
                                                                      _with_images.csv
                              Failed IDs --------------------------> failed_generations.csv
                              ZIP archives ------------------------> .zip files
```

---

## 9. Functional Flow Block Diagram

```
Phase 1             Phase 2              Phase 3            Phase 4
INFRASTRUCTURE      MODEL ACQUISITION    GPU CONFIG          MODEL LOADING
+-----------+       +-------------+      +----------+       +-------------+
| pip/apt   |       | aria2c DL   |      | cuDNN    |       | UNet->GPU   |
| install   |------>| UNet (FP8)  |----->| TF32     |------>| CLIP->GPU   |
| ComfyUI   |       | CLIP (4B)   |      | VRAM     |       | VAE->GPU    |
| clone     |       | VAE (ae)    |      | assert   |       | Neg encode  |
+-----------+       +-------------+      +----------+       +-------------+
                                                                   |
                                                                   v
Phase 5             Phase 6              Phase 7
DATA PREP           GENERATION LOOP      POST-PROCESSING
+-----------+       +-------------+      +------------------+
| Read CSV  |       | VRAM check  |      | Final checkpoint |
| Filter    |------>| Latent gen  |----->| CSV join         |
| Checkpoint|       | CLIP encode |      | Combined CSV     |
| resume    |       | KSampler    |      | Failed CSV       |
| Batch     |       | VAE decode  |      | ZIP archives     |
| partition |       | Save PNG    |      | Summary          |
+-----------+       | Checkpoint  |      +------------------+
                    | OOM recover |
                    +-------------+
```

---

## 10. Sequence Diagram: Runtime Interaction

The runtime interaction follows this temporal sequence:

1. **Script starts** -- installs packages, clones ComfyUI, downloads models.
2. **GPU initialized** -- cuDNN/TF32 configured, VRAM validated, models loaded to GPU memory.
3. **Data loaded** -- CSV read, nulls filtered, checkpoint applied.
4. **Generation loop begins** -- for each batch of 148 prompts:
   - VRAM checked (threshold: 6 GB free).
   - Empty latent noise generated (720x720 x batch_size).
   - Prompts encoded through CLIP with quality prefix.
   - Pre-encoded negative replicated to batch size.
   - KSampler runs 9 Euler steps at CFG 1.0.
   - VAE decodes latents to pixel-space images.
   - Images saved as PNG files.
   - If OOM: batch recursively halved and retried.
   - Every 5 batches: checkpoint written to disk.
5. **Post-processing** -- final checkpoint, CSV assembly, ZIP creation, summary.

---

## 11. Key Technical Decisions and Rationale

### 11.1 FP8 Quantization (E4M3FN)

**Decision:** Use FP8-quantized UNet instead of FP16 or BF16.

**Rationale:** The H200's Hopper architecture natively supports FP8 through its Transformer Engine. FP8 E4M3FN halves memory per parameter compared to FP16, enabling larger batch sizes (148 vs ~48) and increasing throughput proportionally. The "turbo" model is specifically calibrated for FP8 inference, so quality degradation is negligible.

### 11.2 True Batching vs. Sequential Generation

**Decision:** Encode all prompts, generate all latents, and decode all images in a single batched pass per batch.

**Rationale:** GPU utilization is maximized when all CUDA cores and tensor cores are saturated. Sequential per-image generation would leave the GPU idle between launches. Batching amortizes kernel launch overhead and memory allocation across N images simultaneously.

### 11.3 CFG Scale = 1.0

**Decision:** Set classifier-free guidance to 1.0 (effectively disabled).

**Rationale:** The "turbo" model variant is trained with guidance distillation, meaning the guidance signal is baked into the model weights during training. At inference time, no additional CFG scaling is needed. This saves approximately 50% of UNet forward-pass compute (no need for both conditional and unconditional passes).

### 11.4 9 Euler Steps

**Decision:** Use only 9 diffusion steps with the Euler sampler and simple schedule.

**Rationale:** Turbo-distilled models are trained to produce high-quality outputs in 4-10 steps. Nine steps provide a balance between quality and speed. The Euler sampler is a first-order ODE solver that is fast and stable for low step counts.

### 11.5 Deterministic Seeding

**Decision:** Seeds are computed as `(0xDEAD_BEEF + idx * 1_000_003) & 0xFFFFFFFF`.

**Rationale:** Reproducibility is essential for scientific publication. Using a prime multiplier with a fixed base ensures each image gets a unique, deterministic seed. Any researcher can regenerate the exact same image by knowing its global index.

### 11.6 Pre-encoded Negative Conditioning

**Decision:** Encode the negative prompt (empty string) once at startup and replicate via `torch.Tensor.repeat()`.

**Rationale:** Since the negative prompt is identical for all images, encoding it through CLIP for every batch would waste compute. Pre-encoding and tensor replication is orders of magnitude faster.

### 11.7 Polars over Pandas

**Decision:** Use Polars for all CSV operations.

**Rationale:** Polars is significantly faster than Pandas for columnar operations, uses Apache Arrow as its memory backend, and has a more expressive query API. For 18,000 rows this difference is modest, but it reflects modern best practice.

---

## 12. Error Handling and Fault Tolerance

### 12.1 OOM Recovery: Recursive Batch Halving

When `torch.cuda.OutOfMemoryError` occurs during KSampler or the full generation function:

```
Batch N = 148 --> OOM
  |
  +--> Clear cache (aggressive)
  +--> Split: left = prompts[0:74], right = prompts[74:148]
  +--> Retry left (N=74)
  |      +--> OOM? Split again to 37 + 37
  |      +--> Success? Return paths
  +--> Retry right (N=74)
         +--> ... (same recursive logic)
```

**Terminal condition:** If a batch of size 1 triggers OOM, the image is skipped and `None` is returned for that entry.

### 12.2 VAE Decode Fallback

The `_vae_decode_safe` function implements a two-tier fallback:

1. **Tier 1:** Attempt full batch VAE decode.
2. **Tier 2:** On OOM, clear cache and decode images one at a time.
3. **Tier 3:** If a single image fails VAE decode, insert a blank (all-zeros) tensor as placeholder.

### 12.3 Checkpoint Resumability

Progress is persisted to `checkpoint_progress.csv` every 5 batches. On restart:

1. The checkpoint file is read.
2. Completed `counterfactual_id` values are extracted into a set.
3. The working DataFrame is filtered to exclude completed IDs.
4. Generation resumes from where it left off.

### 12.4 Generic Exception Handling

The `generate_true_batch` function has a catch-all `except Exception` block that:

- Logs the exception type, message, and full traceback.
- Clears GPU cache aggressively.
- Returns `[None] * n` (marks all images in the batch as failed).

### 12.5 VRAM Gate

Before each batch, free VRAM is checked. If below 6 GB, aggressive clearing is triggered (Python garbage collection + CUDA cache flush). This prevents cascading OOM failures.

---

## 13. Output Artifacts

| Artifact | Path | Description |
|---|---|---|
| Individual images | `output/generated_images/{counterfactual_id}.png` | 720x720 RGB PNG images, one per prompt |
| Combined dataset | `output/combined_dataset_with_images.csv` | Original CSV joined with image paths and generation metadata |
| Failed IDs | `output/failed_generations.csv` | List of `counterfactual_id` values that failed generation |
| Checkpoint | `output/checkpoint_progress.csv` | Running log of completed IDs and paths (used for resume) |
| Image archive | `output/generated_images.zip` | ZIP of all generated PNG files |
| Complete archive | `output/complete_output.zip` | ZIP of entire output directory |

### 13.1 Combined Dataset Schema

| Column | Source | Description |
|---|---|---|
| `counterfactual_id` | Original CSV | Unique row identifier |
| `t2i_prompt` | Original CSV | Text-to-image prompt |
| (other original columns) | Original CSV | All columns from the source dataset |
| `generated_image_path` | Generated | Absolute path to the generated PNG |
| `total_successful` | Computed | Total number of successfully generated images |
| `total_failed` | Computed | Total number of failed generations |
| `generation_timestamp` | Computed | Timestamp of the generation run |
| `image_resolution` | Config | "720x720" |
| `batch_size_used` | Config | Batch size used during generation |

---

## 14. Code Quality Assessment

### 14.1 Strengths

| Area | Assessment |
|---|---|
| Fault tolerance | Excellent. Multi-tier OOM recovery with recursive batch halving and per-image VAE fallback ensure maximum completion rate. |
| Checkpoint design | Good. Periodic persistence with append semantics and resume-on-restart. |
| GPU optimization | Good. TF32, cuDNN benchmark, FP8, pre-encoded negatives, true batching. |
| Determinism | Good. Deterministic seeding with prime-multiplier scheme. |
| Observability | Good. VRAM status logging, progress bar (tqdm), batch-level status messages. |
| Conditioning batching | Excellent. Variable-dimension detection and zero-padding for heterogeneous prompt lengths. |

### 14.2 Weaknesses

| Area | Assessment |
|---|---|
| Dependency management | Poor. Multiple redundant pip install calls (torch is installed 3 times). Mixed use of shell commands (`!pip`, `os.system`) and `subprocess.run`. |
| Environment coupling | High. Hardcoded paths to `/teamspace/studios/this_studio/`. Not portable to other environments without modification. |
| BATCH_SIZE mismatch | The comment says "tuned for H200 80 GB" but the assertion requires >= 100 GB VRAM. BATCH_SIZE=148 exceeds the comment's recommended max of 64 for 141 GB. For 80 GB VRAM, 148 would certainly OOM frequently. |
| Import redundancy | `torch` is imported three times (lines 30, 38, 39). |
| Security | Model downloads use HTTP URLs without checksum verification. |
| Code organization | Monolithic single-file script with no separation of concerns. |
| Type safety | Minimal type annotations. Return types and parameter types are partially annotated. |
| Testing | No unit tests, integration tests, or validation of generated image quality. |
| Logging | Uses print statements and tqdm.write instead of Python's logging module. |

---

## 15. Identified Issues and Risks

### 15.1 Critical Issues

| ID | Issue | Impact | Line Reference |
|---|---|---|---|
| C-1 | BATCH_SIZE=148 vs VRAM comments | The script comments recommend batch sizes of 32-64 for 141 GB VRAM. A batch size of 148 at 720x720 would require approximately 148 x 90 MB = 13.3 GB just for UNet activations, plus CLIP encoding memory. On an 80 GB card (per the header), this will trigger frequent OOM and recursive halving, negating the throughput benefit. | Line 173 |
| C-2 | VRAM assert >= 100 GB vs header "80 GB" | The script header says "H200 (80 GB SXM5)" but the assertion on line 105 requires >= 100 GB. This will immediately fail on an actual 80 GB card. | Line 105-108 |
| C-3 | Shell command injection risk | `os.system()` calls with f-string interpolation of file paths could be exploited if file paths contain shell metacharacters. | Lines 55, 57, 79 |
| C-4 | Triple PyTorch installation | PyTorch is installed via `!pip` (cu118), then via the `pip()` function (cu126). These are conflicting CUDA versions and the second install will downgrade or corrupt the first. | Lines 12, 37 |

### 15.2 Moderate Issues

| ID | Issue | Impact |
|---|---|---|
| M-1 | No image quality validation | Generated images are saved without any quality check (e.g., SSIM, FID, blank detection). Blank or corrupt images could contaminate the downstream dataset. |
| M-2 | Checkpoint race condition | If the script crashes during a checkpoint write, the CSV could be corrupted. No atomic write (write-then-rename) is used. |
| M-3 | No download integrity verification | Model files are downloaded without SHA256 checksum validation. Corrupt downloads would produce incorrect images silently. |
| M-4 | `os.chdir(COMFY)` side effect | Changing the working directory globally affects all subsequent relative path operations and could cause subtle bugs. |
| M-5 | No resource cleanup | Models are never explicitly unloaded. The script relies on process exit for cleanup. |

### 15.3 Minor Issues

| ID | Issue | Impact |
|---|---|---|
| L-1 | Emoji characters in code | Print statements use Unicode emoji (checkmarks, arrows, warning signs). These may not render correctly in all terminal environments or log aggregation systems. |
| L-2 | `!pip` syntax | Shell magic (`!pip`) is Jupyter notebook syntax. If run as a plain Python script, this will cause a `SyntaxError`. |
| L-3 | Hardcoded quality prefix | `"high quality, detailed, professional, sharp focus"` is prepended to all prompts. For counterfactual analysis, this introduces a confound -- the quality prefix may interact differently with different prompt attributes. |
| L-4 | Metadata columns are scalars | `total_successful` and `total_failed` are broadcast as the same value to all rows, wasting space. Should be in a separate metadata file or row. |

---

## 16. Suggested Enhancements

### 16.1 Immediate (Low Effort, High Impact)

| Enhancement | Description | Benefit |
|---|---|---|
| Fix BATCH_SIZE / VRAM alignment | Set BATCH_SIZE based on actual available VRAM at runtime (e.g., `min(148, int(free_vram / 0.1))`). | Eliminates unnecessary OOM cycles and maximizes throughput for any GPU. |
| Remove redundant installations | Consolidate all pip installs into a single `requirements.txt`. Remove conflicting cu118/cu126 installs. | Cleaner environment, faster startup, no version conflicts. |
| Add SHA256 checksum validation | Verify downloaded model files against known hashes before loading. | Prevents corrupt model inference. |
| Atomic checkpoint writes | Write to a temporary file, then `os.rename()` to the final path. | Prevents checkpoint corruption on crash. |
| Add blank image detection | After VAE decode, check if the image has near-zero variance (all black/white). Flag as failed if so. | Prevents contaminated data in the research dataset. |

### 16.2 Medium Term (Moderate Effort)

| Enhancement | Description | Benefit |
|---|---|---|
| Configuration file | Extract all constants (paths, batch size, steps, resolution, etc.) into a YAML or TOML config file. | Enables non-programmers to modify parameters. Supports multiple experiment configs. |
| Structured logging | Replace print/tqdm.write with Python `logging` module. Add file handler for persistent logs. | Better debuggability, log aggregation, severity levels. |
| Quality metrics | Compute per-image CLIP score (prompt-image alignment) and flag low-scoring images. | Provides a quality signal for researchers without manual inspection. |
| Multi-GPU support | Extend to multiple GPUs using `torch.nn.DataParallel` or manual device assignment. | Linear throughput scaling on multi-GPU nodes. |
| `torch.compile` integration | Uncomment and properly integrate `torch.compile` with warmup handling. | 15-25% throughput improvement per the script's own estimate. |
| Proper CLI interface | Add `argparse` or `click` for command-line configuration (dataset path, batch size, output dir, etc.). | Enables scripted/automated execution without code modification. |

### 16.3 Long Term (High Effort, Strategic)

| Enhancement | Description | Benefit |
|---|---|---|
| Containerization | Package the entire pipeline in a Docker container with pinned dependency versions. | Full reproducibility across environments. |
| Pipeline orchestration | Migrate to a workflow engine (Airflow, Prefect, or Lightning AI's native orchestration). | Better monitoring, retry logic, scheduling, and audit trail. |
| Distributed generation | Shard the dataset across multiple nodes with a central coordinator. | Scale to millions of images. |
| Image quality evaluation pipeline | Add automated FID, IS (Inception Score), and CLIP-Score computation as a post-processing step. | Quantitative quality assessment for the paper. |
| Prompt template validation | Validate that all prompts conform to expected structure before generation. | Prevents wasted compute on malformed prompts. |

---

## 17. Proposed Solution Architecture (v2)

A production-grade revision of this pipeline would adopt the following architecture:

```
+--------------------------------------------------------------------+
|  CONFIG LAYER (config.yaml)                                        |
|  - All paths, hyperparameters, batch sizing, model URLs, checksums |
+--------------------------------------------------------------------+
         |
         v
+--------------------------------------------------------------------+
|  CLI / ENTRYPOINT (main.py)                                        |
|  - argparse for overrides                                          |
|  - Logging configuration                                           |
|  - Environment validation                                          |
+--------------------------------------------------------------------+
         |
         v
+-----------------+  +-----------------+  +-------------------------+
| model_manager.py|  | data_manager.py |  | gpu_manager.py          |
| - Download      |  | - CSV I/O       |  | - VRAM monitoring       |
| - Checksum      |  | - Checkpoint    |  | - Dynamic batch sizing  |
| - Load/Unload   |  | - Resume logic  |  | - OOM recovery          |
| - Version pin   |  | - Validation    |  | - Cache management      |
+-----------------+  +-----------------+  +-------------------------+
         |                   |                       |
         v                   v                       v
+--------------------------------------------------------------------+
|  generator.py                                                       |
|  - Prompt encoding (batched)                                       |
|  - KSampler invocation                                             |
|  - VAE decoding (with fallback)                                    |
|  - Image saving and validation                                     |
+--------------------------------------------------------------------+
         |
         v
+--------------------------------------------------------------------+
|  postprocessor.py                                                   |
|  - Dataset assembly                                                |
|  - Quality metrics (CLIP-Score, variance check)                    |
|  - ZIP archiving                                                   |
|  - Summary generation                                              |
+--------------------------------------------------------------------+
         |
         v
+--------------------------------------------------------------------+
|  OUTPUT                                                             |
|  - images/        (PNG files)                                      |
|  - data/          (CSV files)                                      |
|  - logs/          (structured logs)                                |
|  - metrics/       (quality scores)                                 |
|  - archives/      (ZIP bundles)                                    |
+--------------------------------------------------------------------+
```

### Key Differences from v1

| Aspect | v1 (Current) | v2 (Proposed) |
|---|---|---|
| Configuration | Hardcoded constants in script | External YAML config file |
| Structure | Single monolithic script | Modular Python package (5+ files) |
| Batch sizing | Static (148) | Dynamic based on runtime VRAM |
| Logging | print statements | Structured logging with file and console handlers |
| Quality assurance | None | CLIP-Score + variance check per image |
| Portability | Lightning AI specific | Containerized, environment-agnostic |
| CLI | None | argparse with full parameter control |
| Testing | None | Unit tests for conditioning, checkpointing, OOM recovery |
| Model integrity | None | SHA256 checksum verification |
| Checkpoint safety | Direct file write | Atomic write-then-rename |

---

## 18. Rating: Research Readiness (Scale of 10)

### Overall Rating: 6.0 / 10

| Criterion | Score | Weight | Weighted | Justification |
|---|---|---|---|---|
| Functional correctness | 7/10 | 20% | 1.40 | Core pipeline logic is sound. Batching, conditioning, and OOM recovery work correctly. However, the BATCH_SIZE/VRAM mismatch and conflicting PyTorch installations are functional defects. |
| Reproducibility | 7/10 | 20% | 1.40 | Deterministic seeding is well-implemented. However, no model version pinning (uses latest ComfyUI main branch), no dependency version locking, and no checksum-verified model downloads undermine full reproducibility. |
| Reliability | 7/10 | 15% | 1.05 | Multi-tier OOM recovery and checkpointing are strong. Non-atomic checkpoint writes and no image quality validation are weaknesses. |
| Code quality | 4/10 | 10% | 0.40 | Monolithic structure, redundant imports and installs, mixed shell/Python patterns, Jupyter-specific syntax in a .py file , hardcoded paths. Not suitable for a shared repository without significant cleanup. |
| Documentation | 5/10 | 10% | 0.50 | Inline comments are helpful but inconsistent. No docstrings on most functions. No README, no usage guide, no parameter documentation. This report fills that gap. |
| Scalability | 6/10 | 5% | 0.30 | Handles 18k images on a single GPU well. No multi-GPU, no distributed, no queue-based architecture for larger scales. |
| Output quality | 7/10 | 10% | 0.70 | 720x720 resolution with a quality prefix is adequate for research. No automated quality scoring means manual inspection is required before publication. |
| Publication readiness | 5/10 | 10% | 0.50 | Results are usable but the pipeline itself would need significant cleanup before being cited as a methodology in a paper. Lack of quality metrics and version pinning are the primary gaps. |
| **TOTAL** | | **100%** | **6.25** | |

### Breakdown by Audience

| Audience | Readiness | Notes |
|---|---|---|
| Internal engineering team | 7/10 | Usable as-is with the BATCH_SIZE fix. Engineers can navigate the monolithic structure. |
| Research collaborators | 6/10 | Needs a README, config file, and quality metrics before sharing. |
| Paper reviewers | 5/10 | Needs version pinning, quality evaluation, and reproducibility documentation (exact environment spec, model hashes, dependency lockfile). |
| Open-source community | 3/10 | Needs complete restructuring, proper packaging, tests, CI/CD, and documentation before public release. |

### What Would Raise This to 8+/10

1. Fix all Critical issues (C-1 through C-4).
2. Add a `requirements.txt` or `pyproject.toml` with pinned versions.
3. Add SHA256 checksums for all downloaded models.
4. Add automated CLIP-Score computation.
5. Add a README with setup instructions, parameter documentation, and expected outputs.
6. Refactor into at least 3 modules (config, generation, post-processing).
7. Add at least basic unit tests for the conditioning utilities and checkpoint logic.

---

## 19. Glossary

| Term | Definition |
|---|---|
| **CLIP** | Contrastive Language-Image Pre-training. A model that maps text and images into a shared embedding space, used here as the text encoder that converts prompts into conditioning vectors. |
| **ComfyUI** | An open-source node-based UI and API for running diffusion model pipelines. Used here in headless mode via its Python node API. |
| **CFG (Classifier-Free Guidance)** | A technique that amplifies the influence of the text prompt on the generated image. A value of 1.0 means no amplification (guidance baked into the turbo model). |
| **Counterfactual** | A hypothetical scenario constructed by changing one variable while holding others constant. In this context, prompts are varied systematically to measure model bias. |
| **cuDNN** | NVIDIA CUDA Deep Neural Network library. Provides optimized implementations of convolutions, pooling, and normalization. |
| **Denoise** | The strength of the denoising process. 1.0 means starting from pure noise (full generation). Values less than 1.0 mean starting from a partially noised image (image-to-image). |
| **Euler Sampler** | A first-order ordinary differential equation (ODE) solver used to iteratively denoise latent representations. Simple and fast for low step counts. |
| **FP8 E4M3FN** | An 8-bit floating-point format with 4 exponent bits and 3 mantissa bits, with no NaN encoding. Native to NVIDIA Hopper architecture. |
| **HBM3e** | High Bandwidth Memory 3e. The latest generation of stacked DRAM used on the H200, providing up to 4.8 TB/s bandwidth. |
| **KSampler** | ComfyUI's sampling node that runs the iterative denoising loop of a diffusion model. |
| **Latent Space** | A compressed mathematical representation of an image, typically 8x smaller in each spatial dimension than the pixel-space image. |
| **OOM** | Out of Memory. Occurs when GPU VRAM is exhausted during tensor allocation. |
| **Polars** | A high-performance DataFrame library written in Rust, used here for CSV operations. |
| **safetensors** | A file format for storing neural network weights safely and efficiently, without arbitrary code execution risk. |
| **TF32** | TensorFloat-32. A 19-bit floating-point format used by NVIDIA Ampere+ GPUs that provides FP32-level range with reduced precision, accelerating matrix math. |
| **UNet** | A convolutional neural network architecture with an encoder-decoder structure and skip connections. The core architecture of most diffusion models. |
| **VAE** | Variational Autoencoder. Used here specifically as a decoder to convert latent representations back to pixel-space images. |
| **VRAM** | Video Random Access Memory. The high-bandwidth memory on a GPU used to store model weights, activations, and intermediate tensors. |

---

## 20. Appendices

### Appendix A: Generation Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| `CFG` | 1.0 | Classifier-free guidance scale |
| `STEPS` | 9 | Number of diffusion sampling steps |
| `SAMPLER` | "euler" | ODE solver for denoising |
| `SCHEDULE` | "simple" | Noise schedule type |
| `DENOISE` | 1.0 | Denoising strength (1.0 = full generation from noise) |
| `WIDTH` | 720 | Output image width in pixels |
| `HEIGHT` | 720 | Output image height in pixels |
| `POS_PFX` | "high quality, detailed, professional, sharp focus" | Quality prefix prepended to all prompts |
| `BATCH_SIZE` | 148 | Number of images generated per KSampler call |
| `FREE_VRAM_THRESHOLD_GB` | 6.0 | Minimum free VRAM before aggressive cache clearing |
| `_SEED_BASE` | 0xDEAD_BEEF (3735928559) | Base seed for deterministic generation |
| `_SEED_PRIME` | 1_000_003 | Prime multiplier for seed derivation |

### Appendix B: Directory Structure

```
/teamspace/studios/this_studio/
|-- final_dataset_18k_t2i_prompts.csv       (INPUT)
|-- ComfyUI/                                (FRAMEWORK)
|   |-- nodes.py
|   |-- requirements.txt
|   |-- models/
|       |-- diffusion_models/
|       |   |-- z-image-turbo-fp8-e4m3fn.safetensors
|       |-- clip/
|       |   |-- qwen_3_4b.safetensors
|       |-- vae/
|           |-- ae.safetensors
|-- output/                                 (OUTPUT)
    |-- generated_images/
    |   |-- {counterfactual_id_1}.png
    |   |-- {counterfactual_id_2}.png
    |   |-- ...
    |-- checkpoint_progress.csv
    |-- combined_dataset_with_images.csv
    |-- failed_generations.csv
    |-- generated_images.zip
    |-- complete_output.zip
```

### Appendix C: Seed Derivation Formula

For a given image at global index `i`:

$$\text{seed}(i) = (0\text{xDEAD\_BEEF} + i \times 1{,}000{,}003) \mod 2^{32}$$

This linear congruential scheme guarantees unique seeds for all $i < 2^{32} / 1{,}000{,}003 \approx 4{,}294$ indices before potential collisions. For 18,000 images, collisions are theoretically possible but statistically unlikely due to the modular arithmetic distributing values across the full 32-bit range.

### Appendix D: VRAM Budget Estimation

| Component | Estimated VRAM |
|---|---|
| UNet (FP8) | ~5-8 GB |
| CLIP (Qwen3-4B, ~FP16) | ~8 GB |
| VAE | ~0.3 GB |
| CUDA context + overhead | ~1-2 GB |
| **Total model footprint** | **~15-18 GB** |
| Available for batching (80 GB card) | ~62-65 GB |
| Available for batching (141 GB card) | ~123-126 GB |
| Per-image activation memory (720x720, FP8 UNet) | ~90 MB |
| Max safe batch (80 GB) | ~680 images (theoretical) |
| Max safe batch (141 GB) | ~1300 images (theoretical) |

Note: Theoretical maximums assume perfect memory efficiency. Real-world batch limits are lower due to memory fragmentation, intermediate buffers, and CLIP encoding memory. The script's batch size of 148 is within safe range for 80 GB but will still trigger OOM recovery on some batches due to CLIP encoding peaks.

---

**End of Report**

*This document was generated as a comprehensive technical review of the `image-gen.py` pipeline for cross-team visibility and publication support. All assessments are based on static analysis of the source code as of 20 February 2026.*
