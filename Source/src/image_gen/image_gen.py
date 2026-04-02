# ============================================================================
# LIGHTNING AI NOTEBOOK — Z-Image-Turbo | H200 (141 GB SXM5) OPTIMIZED
# ============================================================================
# Paths:  /teamspace/studios/this_studio/
# GPU:    1× H200 141 GB  (sm_90, FP8, FlashAttn3, 4.8 TB/s HBM3e)
# ============================================================================

import subprocess, sys

# ── Install dependencies ─────────────────────────────────────────────────────
# IMPORTANT: polars/pillow/tqdm must come from PyPI (not the torch CUDA index).
# Install them FIRST in a separate call, then torch from the CUDA wheel index.

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

# Step 1 — Pure-Python / PyPI packages  (NO --index-url here)
pip("polars", "pillow", "tqdm")

# Step 2 — PyTorch with CUDA 12.8 wheels
pip("torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128")

# Step 3 — System tools
import os
os.system("sudo apt -y install -qq aria2 git")

import sys, gc, shutil, traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import polars as pl

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU ok  : {torch.cuda.is_available()}")
# ── Studio root ──────────────────────────────────────────────────────────────
STUDIO  = Path("/teamspace/studios/this_studio")
COMFY   = STUDIO / "ComfyUI"
OUT     = STUDIO / "output"
IMG_OUT = OUT / "generated_images"

for d in [COMFY, IMG_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── Clone ComfyUI ────────────────────────────────────────────────────────────
if not (COMFY / "nodes.py").exists():
    os.system(f"git clone -q https://github.com/comfyanonymous/ComfyUI {COMFY}")
else:
    print("ComfyUI already present — skipping clone.")

os.chdir(COMFY)
os.system(f"{sys.executable} -m pip install -q -r {COMFY}/requirements.txt")

# ── Model dirs ───────────────────────────────────────────────────────────────
(COMFY / "models/diffusion_models").mkdir(parents=True, exist_ok=True)
(COMFY / "models/clip").mkdir(parents=True, exist_ok=True)
(COMFY / "models/vae").mkdir(parents=True, exist_ok=True)

UNET_PATH = COMFY / "models/diffusion_models/z-image-turbo-fp8-e4m3fn.safetensors"
CLIP_PATH  = COMFY / "models/clip/qwen_3_4b.safetensors"
VAE_PATH   = COMFY / "models/vae/ae.safetensors"

def aria2c_dl(url: str, dest: Path):
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return
    print(f"  ↓ Downloading {dest.name}...")
    os.system(
        f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "
        f'"{url}" -d "{dest.parent}" -o "{dest.name}"'
    )

print("Downloading models (skips if cached)...")
aria2c_dl(
    "https://huggingface.co/T5B/Z-Image-Turbo-FP8/resolve/main/z-image-turbo-fp8-e4m3fn.safetensors",
    UNET_PATH,
)
aria2c_dl(
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
    CLIP_PATH,
)
aria2c_dl(
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    VAE_PATH,
)
print("Models ready!")

# ── H200 global torch settings ───────────────────────────────────────────────
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

DEVICE = "cuda:0"
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"CUDA: {torch.cuda.get_device_name(0)}  |  VRAM: {total_vram_gb:.1f} GB")
assert total_vram_gb >= 100, (
    f"Expected H200 (>=100 GB) but got {total_vram_gb:.1f} GB — "
    "adjust BATCH_SIZE below for your actual GPU."
)

# ── ComfyUI node classes ──────────────────────────────────────────────────────
sys.path.insert(0, str(COMFY))
from nodes import NODE_CLASS_MAPPINGS

print("Loading nodes...")
UNETLoader       = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader       = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader        = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode   = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler         = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode        = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models onto GPU...")
with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="wan")[0]
    vae  = VAELoader.load_vae("ae.safetensors")[0]

print("✓ Models loaded!")

# ============================================================================
# VRAM HELPERS
# ============================================================================

def vram_free_gb(device: int = 0) -> float:
    return torch.cuda.mem_get_info(device)[0] / 1024**3

def print_gpu_status():
    free  = vram_free_gb()
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    used  = total - free
    print(f"\n  H200: {used:.1f}/{total:.1f} GB used  ({free:.1f} GB free)\n")

def clear_cache(aggressive: bool = False):
    if aggressive:
        gc.collect()
    torch.cuda.empty_cache()
    if aggressive:
        gc.collect()

FREE_VRAM_THRESHOLD_GB = 6.0

# ============================================================================
# GENERATION CONFIG
# ============================================================================

CFG      = 1.0
STEPS    = 9
SAMPLER  = "euler"
SCHEDULE = "simple"
DENOISE  = 1.0
WIDTH    = 720
HEIGHT   = 720

# ── Positive quality prefix ───────────────────────────────────────────────────
# Prepended to every row's t2i_prompt.
POS_PFX = (
    "8k uhd, ultra high resolution, photorealistic, realistic, "
    "RAW photo, masterpiece, best quality, high quality, detailed, "
    "professional, sharp focus, intricate details, natural lighting, "
    "cinematic composition, depth of field"
)

# ── Anti-text / anti-sign block ───────────────────────────────────────────────
# Forcibly appended to EVERY image's negative prompt regardless of what the
# CSV row specifies. Covers every form of text/sign/poster a model might render.
NO_TEXT_NEG = (
    "text, written text, letters, words, numbers, typography, font, "
    "readable text, printed text, handwriting, graffiti, "
    "holding sign, holding poster, holding board, holding banner, holding placard, "
    "label, sticker, badge, caption, subtitle, watermark, logo, "
    "speech bubble, thought bubble, comic text"
)

# ── General quality negative suffix ──────────────────────────────────────────
# Appended after the row-level content and NO_TEXT_NEG.
QUALITY_NEG = (
    "low quality, worst quality, blurry, out of focus, noisy, grainy, "
    "pixelated, deformed, distorted, disfigured, bad anatomy, "
    "ugly, duplicate, watermark, signature, "
    "cartoon, anime, illustration, painting, drawing, sketch, "
    "oversaturated, overexposed, underexposed, flat lighting, "
    "low resolution, jpeg artifacts, compression artifacts"
)

# Full fallback used when a row has no t2i_negative_prompt value
NEG_FALLBACK = f"{NO_TEXT_NEG}, {QUALITY_NEG}"

BATCH_SIZE = 256

CHECKPOINT_PATH = OUT / "checkpoint_progress.csv"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH = STUDIO / "Hate-race.csv"

COL_ID         = "counterfactual_id"    # unique row identifier → filename
COL_PROMPT     = "t2i_prompt"           # positive prompt column
COL_NEG_PROMPT = "t2i_negative_prompt"  # per-row negative prompt column

# ── Deterministic seeds ───────────────────────────────────────────────────────
_SEED_BASE  = 0xDEAD_BEEF
_SEED_PRIME = 1_000_003

def image_seed(global_idx: int) -> int:
    return (_SEED_BASE + global_idx * _SEED_PRIME) & 0xFFFF_FFFF

# ============================================================================
# CONDITIONING UTILITIES
# ============================================================================

def find_variable_dim(tensors: list[torch.Tensor]) -> int | None:
    if len(tensors) <= 1:
        return None
    for dim in range(tensors[0].ndim):
        if len({t.shape[dim] for t in tensors}) > 1:
            return dim
    return None


def pad_to_max(tensors: list[torch.Tensor], dim: int) -> list[torch.Tensor]:
    max_size = max(t.shape[dim] for t in tensors)
    out = []
    for t in tensors:
        diff = max_size - t.shape[dim]
        if diff == 0:
            out.append(t)
            continue
        pad_arg = [0] * (2 * t.ndim)
        pad_arg[2 * (t.ndim - 1 - dim) + 1] = diff
        out.append(F.pad(t, pad_arg, mode="constant", value=0))
    return out


def stack_conditionings(cond_list: list) -> list:
    if len(cond_list) == 1:
        return cond_list[0]

    batched = []
    for elem_idx in range(len(cond_list[0])):
        main_tensors = [c[elem_idx][0] for c in cond_list]
        dicts        = [c[elem_idx][1] for c in cond_list]

        var_dim = find_variable_dim(main_tensors)
        if var_dim is not None:
            main_tensors = pad_to_max(main_tensors, var_dim)
        batched_main = torch.cat(main_tensors, dim=0)

        all_keys = set(dicts[0].keys())
        for d in dicts[1:]:
            all_keys &= set(d.keys())

        merged_dict: dict = {}
        for key in all_keys:
            vals = [d[key] for d in dicts]
            if all(v is None for v in vals):
                merged_dict[key] = None
                continue
            ref  = next(v for v in vals if v is not None)
            vals = [torch.zeros_like(ref) if v is None else v for v in vals]
            if not torch.is_tensor(vals[0]):
                merged_dict[key] = vals[0]
                continue
            var_dim_k = find_variable_dim(vals)
            if var_dim_k is not None:
                vals = pad_to_max(vals, var_dim_k)
            try:
                merged_dict[key] = torch.cat(vals, dim=0)
            except RuntimeError as e:
                tqdm.write(f"  ⚠ Skipping key '{key}': {e}")
                merged_dict[key] = vals[0]

        batched.append([batched_main, merged_dict])
    return batched


@torch.inference_mode()
def encode_prompts(prompts: list[str], prefix: str = "") -> list:
    """Encode positive prompts, prepending a quality prefix."""
    cond_list = []
    for p in prompts:
        full = f"{prefix}, {p}" if prefix else p
        cond_list.append(CLIPTextEncode.encode(clip, full)[0])
    return stack_conditionings(cond_list)


@torch.inference_mode()
def encode_neg_prompts(neg_prompts: list[str | None]) -> list:
    """
    Build and encode per-row negative prompts.

    Final negative for each row = <CSV row neg> + NO_TEXT_NEG + QUALITY_NEG

    NO_TEXT_NEG is ALWAYS included — it suppresses signs, posters, banners,
    placards, boards with text and any readable characters in the image,
    regardless of what the CSV row specifies.

    If a row has no CSV negative, NEG_FALLBACK (= NO_TEXT_NEG + QUALITY_NEG)
    is used as the base so no row ever lacks anti-text guidance.
    """
    cond_list = []
    for neg in neg_prompts:
        if neg and str(neg).strip():
            row_neg  = str(neg).strip().rstrip(",")
            full_neg = f"{row_neg}, {NO_TEXT_NEG}, {QUALITY_NEG}"
        else:
            full_neg = NEG_FALLBACK
        cond_list.append(CLIPTextEncode.encode(clip, full_neg)[0])
    return stack_conditionings(cond_list)


# ============================================================================
# GENERATION — TRUE BATCHING WITH H200-AWARE OOM HANDLING
# ============================================================================

@torch.inference_mode()
def _vae_decode_safe(vae, samples) -> torch.Tensor:
    """Batch VAE decode with per-image fallback. Returns [N,H,W,C] CPU float32."""
    try:
        return VAEDecode.decode(vae, samples)[0].detach().cpu()
    except torch.cuda.OutOfMemoryError:
        tqdm.write("  ⚠ VAE batch OOM — falling back to per-image decode")
        clear_cache(aggressive=True)
        results = []
        for i in range(samples["samples"].shape[0]):
            single = {"samples": samples["samples"][i:i+1]}
            try:
                img = VAEDecode.decode(vae, single)[0].detach().cpu()
            except torch.cuda.OutOfMemoryError:
                tqdm.write(f"  ✗ VAE OOM on image {i} — inserting blank")
                img = torch.zeros(1, HEIGHT, WIDTH, 3)
            results.append(img)
            clear_cache()
        return torch.cat(results, dim=0)


@torch.inference_mode()
def generate_true_batch(
    prompts: list[str],
    neg_prompts: list[str | None],
    ids: list[str],
    global_start_idx: int = 0,
) -> list[str | None]:
    """
    Generate one batch.  OOM → halves batch recursively.
    Returns list of output paths (None on failure).
    """
    n = len(prompts)
    try:
        free = vram_free_gb()
        if free < FREE_VRAM_THRESHOLD_GB:
            tqdm.write(f"  ⚠ Low VRAM ({free:.1f} GB), clearing...")
            clear_cache(aggressive=True)

        # 1. Latent batch
        latent_batch = EmptyLatentImage.generate(WIDTH, HEIGHT, batch_size=n)[0]

        # 2. Positive conditioning
        pos_batch = encode_prompts(prompts, prefix=POS_PFX)

        # 3. Negative conditioning — per-row + NO_TEXT_NEG + QUALITY_NEG always
        neg_batch = encode_neg_prompts(neg_prompts)

        # 4. KSampler
        seed    = image_seed(global_start_idx)
        samples = KSampler.sample(
            unet, seed, STEPS, CFG, SAMPLER, SCHEDULE,
            pos_batch, neg_batch, latent_batch, denoise=DENOISE,
        )[0]

        # 5. VAE decode
        decoded = _vae_decode_safe(vae, samples)   # [N,H,W,C] CPU float32

        # 6. Save
        paths = []
        for i, row_id in enumerate(ids):
            img_np = (decoded[i].clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
            fpath  = IMG_OUT / f"{row_id}.png"
            Image.fromarray(img_np).save(fpath)
            paths.append(str(fpath))

        tqdm.write(f"  ✓ {n} imgs  |  {vram_free_gb():.1f} GB free")
        return paths

    except torch.cuda.OutOfMemoryError:
        tqdm.write(f"  ✗ OOM on batch={n} — halving...")
        clear_cache(aggressive=True)
        if n == 1:
            tqdm.write("  ✗ OOM on single image — skipping")
            return [None]
        mid   = n // 2
        left  = generate_true_batch(
            prompts[:mid], neg_prompts[:mid], ids[:mid],
            global_start_idx,
        )
        right = generate_true_batch(
            prompts[mid:], neg_prompts[mid:], ids[mid:],
            global_start_idx + mid,
        )
        return left + right

    except Exception as e:
        tqdm.write(f"  ✗ Unexpected error: {type(e).__name__}: {e}")
        tqdm.write(traceback.format_exc())
        clear_cache(aggressive=True)
        return [None] * n

# ============================================================================
# DATASET + CHECKPOINTING
# ============================================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\nExpected: {DATASET_PATH}"
    )

df = (
    pl.read_csv(str(DATASET_PATH))
    .filter(pl.col(COL_PROMPT).is_not_null())
)

# Ensure the negative prompt column exists; auto-create as null if absent
if COL_NEG_PROMPT not in df.columns:
    print(f"⚠ Column '{COL_NEG_PROMPT}' not found — using NEG_FALLBACK for all rows")
    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(COL_NEG_PROMPT))
else:
    neg_missing = df[COL_NEG_PROMPT].null_count()
    print(f"✓ Negative prompt column found  |  {neg_missing} null rows → NEG_FALLBACK")

print(f"✓ Dataset: {len(df)} rows  |  columns: {df.columns}")

start_offset = 0
if CHECKPOINT_PATH.exists():
    try:
        ckpt_df      = pl.read_csv(str(CHECKPOINT_PATH))
        done_ids     = set(ckpt_df[COL_ID].to_list())
        start_offset = len(done_ids)
        df           = df.filter(~pl.col(COL_ID).is_in(done_ids))
        print(f"✓ Checkpoint: {start_offset} done — {len(df)} remaining")
    except Exception as e:
        print(f"⚠ Checkpoint read error ({e}), starting fresh")
else:
    print("✓ Fresh start — no checkpoint")

# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

total_rows         = len(df)
successful_count   = 0
failed_ids         = []
batch_counter      = 0
global_img_counter = start_offset
pending_ckpt       = []

print(f"\n{'='*60}")
print(f"H200 TRUE-BATCH GENERATION  (per-row negatives + anti-text)")
print(f"  Images   : {total_rows}")
print(f"  Batch sz : {BATCH_SIZE}")
print(f"  Steps    : {STEPS}  |  {WIDTH}×{HEIGHT}")
print(f"{'='*60}\n")

print_gpu_status()

pbar = tqdm(total=total_rows, desc="Generating", unit="img")

for batch_start in range(0, total_rows, BATCH_SIZE):
    batch_df    = df.slice(batch_start, BATCH_SIZE)
    prompts     = batch_df[COL_PROMPT].to_list()
    neg_prompts = batch_df[COL_NEG_PROMPT].to_list()
    ids         = batch_df[COL_ID].to_list()

    paths = generate_true_batch(
        prompts,
        neg_prompts,
        ids,
        global_start_idx=global_img_counter,
    )

    for row_id, path in zip(ids, paths):
        if path is not None:
            successful_count += 1
            pending_ckpt.append({COL_ID: row_id, "image_path": path})
        else:
            failed_ids.append(row_id)

    batch_counter      += 1
    global_img_counter += len(ids)
    pbar.update(len(ids))

    clear_cache()

    # Checkpoint every 5 batches
    if batch_counter % 5 == 0 and pending_ckpt:
        new_df = pl.DataFrame(pending_ckpt)
        if CHECKPOINT_PATH.exists():
            new_df = pl.concat([pl.read_csv(str(CHECKPOINT_PATH)), new_df])
        new_df.write_csv(str(CHECKPOINT_PATH))
        pending_ckpt.clear()
        total_done = successful_count + len(failed_ids)
        rate       = (successful_count / total_done * 100) if total_done else 0.0
        tqdm.write(f"\n  ✓ Checkpoint @{batch_counter} — {successful_count} ok / {rate:.1f}%\n")

    if batch_counter % 10 == 0:
        pbar.clear()
        print_gpu_status()

# Final checkpoint flush
if pending_ckpt:
    new_df = pl.DataFrame(pending_ckpt)
    if CHECKPOINT_PATH.exists():
        new_df = pl.concat([pl.read_csv(str(CHECKPOINT_PATH)), new_df])
    new_df.write_csv(str(CHECKPOINT_PATH))
    tqdm.write(f"✓ Final checkpoint flush: {len(pending_ckpt)} rows")

pbar.close()

# ============================================================================
# FINAL DATASET ASSEMBLY
# ============================================================================

df_orig = pl.read_csv(str(DATASET_PATH))
mapping = [
    {COL_ID: f.stem, "generated_image_path": str(IMG_OUT / f.name)}
    for f in IMG_OUT.glob("*.png")
]

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
df_final = (
    df_orig
    .join(pl.DataFrame(mapping), on=COL_ID, how="left")
    .with_columns([
        pl.lit(successful_count).alias("total_successful"),
        pl.lit(len(failed_ids)).alias("total_failed"),
        pl.lit(ts).alias("generation_timestamp"),
        pl.lit(f"{WIDTH}x{HEIGHT}").alias("image_resolution"),
        pl.lit(BATCH_SIZE).alias("batch_size_used"),
    ])
)
final_csv = OUT / "combined_dataset_with_images.csv"
df_final.write_csv(str(final_csv))
print(f"✓ Final CSV: {final_csv}")

if failed_ids:
    fail_csv = OUT / "failed_generations.csv"
    pl.DataFrame({COL_ID: failed_ids, "status": "failed"}).write_csv(str(fail_csv))
    print(f"⚠ Failed IDs: {fail_csv}")

# ── Compress outputs ──────────────────────────────────────────────────────────
import zipfile

images_zip = OUT / "generated_images.zip"
print(f"\nCompressing images → {images_zip.name} ...")
with zipfile.ZipFile(str(images_zip), "w",
                     compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as zf:
    for img_file in sorted(IMG_OUT.glob("*.png")):
        zf.write(str(img_file), arcname=img_file.name)
print(f"✓ {images_zip.name} ({images_zip.stat().st_size / 1024**2:.1f} MB)")

try:
    shutil.rmtree(str(IMG_OUT))
    print(f"✓ Removed raw image folder: {IMG_OUT.name}")
except Exception as _e:
    print(f"⚠ Could not remove image folder: {_e}")

complete_zip = OUT / "complete_output.zip"
output_files = sorted(
    f for f in OUT.iterdir()
    if f.is_file() and f.name != complete_zip.name
)
print(f"\nAssembling {complete_zip.name} ({len(output_files)} file(s)) ...")
with zipfile.ZipFile(str(complete_zip), "w",
                     compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as zf:
    for fpath in output_files:
        zf.write(str(fpath), arcname=fpath.name)
        print(f"  + {fpath.name}")
print(f"✓ {complete_zip.name} ({complete_zip.stat().st_size / 1024**2:.1f} MB)")

# ── Summary ───────────────────────────────────────────────────────────────────
total_att    = successful_count + len(failed_ids)
success_rate = (successful_count / total_att * 100) if total_att else 0.0

print(f"\n{'='*60}  SUMMARY")
print(f"  Total attempted  : {total_att}")
print(f"  Successful       : {successful_count}")
print(f"  Failed           : {len(failed_ids)}")
print(f"  Success rate     : {success_rate:.2f}%")
print(f"  Resolution       : {WIDTH}×{HEIGHT}")
print(f"  Batch size       : {BATCH_SIZE} images / KSampler call")
print(f"{'='*60}")

print_gpu_status()
print("✓ DONE!")