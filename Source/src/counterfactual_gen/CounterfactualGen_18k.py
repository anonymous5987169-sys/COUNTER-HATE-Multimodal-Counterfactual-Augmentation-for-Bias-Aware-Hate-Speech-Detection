# %% [markdown]
# ---Note: This is the 18k dataset generation script only for KAGGLE NOTEBOOK. 
# ## Cell 2 — Verify environment

# %%
import numpy as np
import torch

assert tuple(int(x) for x in np.__version__.split(".")[:2]) < (2, 0), \
    f"numpy {np.__version__} detected — must be <2.0. Re-run Cell 1."

print(f"✅ numpy  {np.__version__}")
print(f"✅ torch  {torch.__version__}  |  CUDA {torch.version.cuda}")

n_gpus = torch.cuda.device_count()
print(f"✅ GPUs   {n_gpus}")
for i in range(n_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"   GPU {i}: {p.name}  ({p.total_memory / 1e9:.1f} GB VRAM)")

assert n_gpus >= 2, "Need T4×2. Go to Settings → Accelerator → GPU T4×2."

# %% [markdown]
# ## Cell 3 — Imports & configuration

# %%
import os, re, json, hashlib, time, logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
INPUT_CSV  = Path("/kaggle/input/datasets/veeraj16/6k-samples-dataset/hate_speech_dataset_6k.csv")
OUTPUT_CSV = Path("/kaggle/working/final_dataset_18k.csv")

# ── Model config ────────────────────────────────────────────────────────────
MODEL_ID            = "Qwen/Qwen2.5-3B-Instruct"

TENSOR_PARALLEL     = 2
MAX_MODEL_LEN       = 1024
MAX_NEW_TOKENS      = 128
TEMPERATURE         = 0.25
TOP_P               = 0.9
GPU_MEM_UTIL        = 0.92

log.info("Config loaded. Model: %s | GPUs: %d", MODEL_ID, TENSOR_PARALLEL)

# %% [markdown]
# ## Cell 4 — Identity dictionaries & pre-compiled regex

# %%
IDENTITY_AXES: dict[str, list[str]] = {
    "race_ethnicity": [
        "Black", "White", "Asian", "Latino", "Hispanic", "Arab",
        "African", "European", "Indian", "Chinese", "Japanese", "Korean",
        "Mexican", "Jewish", "Native American", "Indigenous", "Pacific Islander",
        "Caribbean", "Middle Eastern", "Southeast Asian", "South Asian",
        "East Asian", "Caucasian", "biracial", "multiracial",
    ],
    "religion": [
        "Muslim", "Christian", "Jewish", "Hindu", "Buddhist", "Sikh",
        "Atheist", "Catholic", "Protestant", "Mormon", "Evangelical",
        "Orthodox", "Sunni", "Shia", "Jew", "Islam", "Islamic",
    ],
    "gender_sexuality": [
        "women", "men", "transgender", "nonbinary", "gay", "lesbian",
        "bisexual", "queer", "straight", "heterosexual", "homosexual",
        "female", "male", "trans", "LGBTQ", "cisgender", "asexual",
    ],
    "nationality": [
        "American", "Mexican", "Chinese", "Indian", "Nigerian", "Brazilian",
        "British", "French", "German", "Russian", "Iranian", "Iraqi",
        "Syrian", "Afghan", "Pakistani", "Somali", "Ethiopian",
        "Japanese", "Korean", "Colombian", "Venezuelan", "Cuban",
    ],
    "disability": [
        "disabled", "blind", "deaf", "autistic", "mentally ill",
        "wheelchair user", "handicapped", "special needs",
    ],
    "age": [
        "old", "young", "elderly", "teenager", "millennial",
        "boomer", "Gen Z", "senior", "youth",
    ],
}

SLUR_TO_IDENTITY: dict[str, str] = {
    "nigger": "Black",   "nigga": "Black",    "niggas": "Black",   "negro": "Black",
    "darkie": "Black",   "coon": "Black",     "spook": "Black",    "negroes": "Black",
    "chink": "Asian",    "gook": "Asian",     "slant": "Asian",    "zipperhead": "Asian",
    "jap": "Japanese",   "nip": "Japanese",
    "spic": "Latino",    "spick": "Latino",   "beaner": "Latino",  "wetback": "Latino",
    "gringo": "White",   "cracker": "White",  "redneck": "White",  "honky": "White",
    "kike": "Jewish",    "yid": "Jewish",     "hebe": "Jewish",
    "muzzie": "Muslim",  "towelhead": "Muslim","raghead": "Muslim", "muzzy": "Muslim",
    "fag": "gay",        "faggot": "gay",     "dyke": "lesbian",   "tranny": "transgender",
    "homo": "homosexual",
    "whore": "women",    "bitch": "women",    "slut": "women",     "cunt": "women",
    "retard": "disabled","retarded": "disabled","cripple": "disabled",
}

TARGET_GROUP_TO_AXIS: dict[str, Optional[str]] = {
    "race/ethnicity":              "race_ethnicity",
    "religion":                    "religion",
    "gender":                      "gender_sexuality",
    "sexual_orientation":          "gender_sexuality",
    "national_origin/citizenship": "nationality",
    "disability":                  "disability",
    "age":                         "age",
    "multiple/none":               None,
}

TERM_TO_AXIS: dict[str, str] = {}
for _ax, _terms in IDENTITY_AXES.items():
    for _t in _terms:
        TERM_TO_AXIS[_t.lower()] = _ax
for _slur, _id in SLUR_TO_IDENTITY.items():
    if (_ax := TERM_TO_AXIS.get(_id.lower())):
        TERM_TO_AXIS[_slur.lower()] = _ax


@dataclass(slots=True)
class DetectedTerm:
    term: str; axis: str; identity: str
    start: int; end: int; is_slur: bool

@dataclass(slots=True)
class Replacement:
    original_term: str; replacement: str
    axis: str; original_identity: str


def _compile_patterns():
    slur_pats, id_pats = [], []
    for slur in sorted(SLUR_TO_IDENTITY, key=len, reverse=True):
        identity = SLUR_TO_IDENTITY[slur]
        axis     = TERM_TO_AXIS.get(identity.lower(), "unknown")
        slur_pats.append((
            re.compile(r'\b' + re.escape(slur) + r'(?:s|es|ed|ing)?\b', re.I),
            identity, axis,
        ))
    for ax, terms in IDENTITY_AXES.items():
        for term in sorted(terms, key=len, reverse=True):
            id_pats.append((
                re.compile(r'\b' + re.escape(term) + r'(?:s|es)?\b', re.I),
                term, ax,
            ))
    return slur_pats, id_pats

_SLUR_PATS, _ID_PATS = _compile_patterns()

_CLEAN_PREFIX_RE = re.compile(
    r'^(?:rewritten[\s\w]*?|output|result|here[\s\w]*?|text|sentence|answer|cf\s*\d*|'
    r'counterfactual[\s\w]*?):\s*',
    re.I,
)
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.S)

log.info("✅ Identity dictionaries & regex patterns ready (%d slur + %d identity patterns)",
         len(_SLUR_PATS), len(_ID_PATS))

# %% [markdown]
# ## Cell 5 — Regex helpers (detection only — used to build LLM prompts)
# NOTE: regex_swap / try_regex_cf are removed. Detection is kept because
# build_explicit_prompt needs to know which terms exist and what to swap them to.

# %%
def detect_identity_terms(text: str) -> list[DetectedTerm]:
    found: list[DetectedTerm] = []
    occupied: set[int] = set()
    lower = text.lower()

    def _claim(s: int, e: int) -> bool:
        span = set(range(s, e))
        if span & occupied: return False
        occupied.update(span); return True

    for pat, identity, axis in _SLUR_PATS:
        for m in pat.finditer(lower):
            if _claim(m.start(), m.end()):
                found.append(DetectedTerm(
                    text[m.start():m.end()], axis, identity,
                    m.start(), m.end(), True,
                ))

    for pat, term, axis in _ID_PATS:
        for m in pat.finditer(lower):
            if _claim(m.start(), m.end()):
                found.append(DetectedTerm(
                    text[m.start():m.end()], axis, term,
                    m.start(), m.end(), False,
                ))

    found.sort(key=lambda d: d.start)
    return found


def _pick_replacement(identity: str, axis: str, cf_index: int, seed: str) -> Optional[str]:
    candidates = [t for t in IDENTITY_AXES.get(axis, []) if t.lower() != identity.lower()]
    if not candidates:
        return None
    h0 = int(hashlib.md5(f"{seed}_0".encode()).hexdigest(), 16) % len(candidates)
    if cf_index == 0:
        return candidates[h0]
    h1 = int(hashlib.md5(f"{seed}_1".encode()).hexdigest(), 16) % len(candidates)
    if len(candidates) > 1 and h1 == h0:
        h1 = (h1 + 1) % len(candidates)
    return candidates[h1]


def _pick_implicit_identity(sample_id: str, target_group: str, cf_index: int) -> tuple[str, str]:
    axis = TARGET_GROUP_TO_AXIS.get(target_group)
    if axis and axis in IDENTITY_AXES:
        terms = IDENTITY_AXES[axis]
        h0 = int(hashlib.md5(f"{sample_id}_0_impl".encode()).hexdigest(), 16) % len(terms)
        if cf_index == 0: return terms[h0], axis
        h1 = int(hashlib.md5(f"{sample_id}_1_impl".encode()).hexdigest(), 16) % len(terms)
        if len(terms) > 1 and h1 == h0: h1 = (h1 + 1) % len(terms)
        return terms[h1], axis
    axes = list(IDENTITY_AXES.keys())
    h = int(hashlib.md5(f"{sample_id}_{cf_index}_noax".encode()).hexdigest(), 16)
    ax = axes[h % len(axes)]
    if cf_index == 1:
        h0 = int(hashlib.md5(f"{sample_id}_0_noax".encode()).hexdigest(), 16)
        if ax == axes[h0 % len(axes)] and len(axes) > 1:
            ax = axes[(h + 1) % len(axes)]
    terms = IDENTITY_AXES[ax]
    return terms[h % len(terms)], ax


def _build_row(sample_id: str, cf_text: str, row: dict, cf_index: int) -> dict:
    return {
        "original_sample_id": sample_id,
        "counterfactual_id":  f"{sample_id}_cf{cf_index + 1}",
        "text":               cf_text,
        "class_label":        row["class_label"],
        "target_group":       row.get("target_group", "multiple/none"),
        "polarity":           row.get("polarity", "non-hate"),
        "hate_score":         None,
        "confidence":         None,
        "cf_type":            f"counterfactual_{cf_index + 1}",
        "t2i_prompt":         "",
    }


def _injection_fallback(text: str, target_identity: str, cf_index: int) -> str:
    words = text.split()
    if len(words) >= 6:
        mid = len(words) // 3
        words.insert(mid, f"({target_identity})")
        return " ".join(words)
    if cf_index == 0:
        return f"{text} [about {target_identity} people]"
    return f"[{target_identity}]: {text}"


log.info("✅ Detection helpers ready")

# %% [markdown]
# ## Cell 6 — Build full LLM queue (all 12k counterfactuals go to the model)
# Phase 1 CPU regex swap is removed. Every row × 2 CFs is sent to the LLM.

# %%
df_source = pl.read_csv(str(INPUT_CSV))
log.info("Source: %d rows | classes: %s",
         len(df_source),
         df_source["class_label"].value_counts().to_dicts())

rows = df_source.to_dicts()

llm_queue: list[dict] = []

for row in rows:
    detected = detect_identity_terms(row["text"])
    for cf_idx in range(2):
        # If explicit identity terms found, tell the LLM exactly what to swap.
        # Otherwise fall back to implicit: ask the LLM to rewrite for a chosen group.
        if detected:
            reps = [
                Replacement(d.term, rep, d.axis, d.identity)
                for d in detected
                if (rep := _pick_replacement(d.identity, d.axis, cf_idx, d.term))
            ]
        else:
            reps = []

        target_identity, axis = _pick_implicit_identity(
            row["sample_id"], row.get("target_group", "multiple/none"), cf_idx
        )

        llm_queue.append({
            "row":             row,
            "cf_index":        cf_idx,
            "target_identity": target_identity,
            "axis":            axis,
            "replacements":    reps,   # non-empty → explicit prompt; empty → implicit prompt
        })

log.info("LLM queue built: %d prompts (%d rows × 2 CFs)", len(llm_queue), len(rows))

# %% [markdown]
# ## Cell 7 — Load vLLM on both T4s

# %%
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

log.info("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

log.info("Loading vLLM engine  (tensor_parallel_size=%d) …", TENSOR_PARALLEL)
t0 = time.time()

llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=TENSOR_PARALLEL,
    dtype="float16",
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEM_UTIL,
    trust_remote_code=True,
    enforce_eager=False,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_NEW_TOKENS,
    stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],
    skip_special_tokens=True,
    repetition_penalty=1.1,
)

log.info("✅ vLLM engine ready in %.1f s", time.time() - t0)

# %% [markdown]
# ## Cell 8 — Prompt engineering

# %%
_SYS_EXPLICIT = (
    "You are a dataset augmentation tool for hate-speech research.\n"
    "You swap identity group names in sentences. Follow instructions exactly.\n"
    "Output ONLY the rewritten sentence. No explanation. No quotes. No prefix."
)

_SYS_IMPLICIT = (
    "You are a dataset augmentation tool for hate-speech research.\n"
    "You rewrite sentences to reference a different identity group.\n"
    "Preserve original meaning, tone, and structure exactly.\n"
    "Output ONLY the rewritten sentence. No explanation. No quotes. No prefix."
)


def _chat_prompt(system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_explicit_prompt(text: str, replacements: list[Replacement]) -> str:
    swaps = "\n".join(f'• "{r.original_term}" → "{r.replacement}"' for r in replacements)
    user = (
        "Swap these identity terms and output ONLY the result.\n\n"
        "Example:\n"
        "Swap: • \"Black\" → \"Asian\"\n"
        "Sentence: All Black people should leave.\n"
        "Output: All Asian people should leave.\n\n"
        f"Swap:\n{swaps}\n"
        f"Sentence: {text}\n"
        "Output:"
    )
    return _chat_prompt(_SYS_EXPLICIT, user)


def build_implicit_prompt(
    text: str,
    target_identity: str,
    axis: str,
    polarity: str,
) -> str:
    tone = "hateful/hostile" if polarity == "hate" else "neutral or counter-speech"
    axis_label = axis.replace("_", " ")
    user = (
        "Example:\n"
        "Rewrite for 'Muslim' (religion), hateful/hostile tone.\n"
        "Sentence: Those people ruin every neighbourhood.\n"
        "Output: Muslims ruin every neighbourhood.\n\n"
        f"Rewrite the sentence so it refers to '{target_identity}' ({axis_label}). "
        f"Keep the {tone} tone. If no group is mentioned, insert a natural reference. "
        f"Output ONLY the rewritten sentence.\n\n"
        f"Sentence: {text}\n"
        "Output:"
    )
    return _chat_prompt(_SYS_IMPLICIT, user)


def clean_output(raw: str, original: str) -> str:
    text = _THINK_BLOCK_RE.sub("", raw).strip()
    text = text.strip('"').strip("'")
    text = _CLEAN_PREFIX_RE.sub("", text).strip().strip('"').strip("'")

    if "\n\n" in text:
        text = text.split("\n\n")[0].strip()
    text = text.strip()

    if not text or len(text) < 4:
        return ""

    ratio = len(text) / max(len(original), 1)
    if not (0.25 <= ratio <= 3.0):
        return ""

    non_ascii = sum(
        1 for c in text
        if ord(c) > 127
        and not (0x1F300 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF)
    )
    if non_ascii / max(len(text), 1) > 0.08:
        return ""

    if re.match(r'^example\s*:', text, re.I):
        return ""

    return text


log.info("✅ Prompt templates ready")

# %% [markdown]
# ## Cell 9 — LLM batched inference (all 12k prompts)

# %%
def run_llm(queue: list[dict]) -> list[dict]:
    if not queue:
        log.info("LLM queue is empty — nothing to do.")
        return []

    log.info("Building %d prompts …", len(queue))
    prompts: list[str] = []

    for item in queue:
        reps      = item["replacements"]
        text      = item["row"]["text"]
        target_id = item["target_identity"]
        axis      = item["axis"]
        polarity  = item["row"].get("polarity", "non-hate")

        # Explicit prompt when the LLM has concrete swap targets;
        # implicit prompt when the text has no detectable identity terms.
        if reps:
            prompts.append(build_explicit_prompt(text, reps))
        else:
            prompts.append(build_implicit_prompt(text, target_id, axis, polarity))

    log.info("Running vLLM on %d prompts across %d GPUs …", len(prompts), TENSOR_PARALLEL)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    log.info("vLLM complete in %.1f s  (%.1f prompts/s)", elapsed, len(prompts) / elapsed)

    results: list[dict] = []
    fallback_count = 0

    for item, output in zip(queue, outputs):
        row       = item["row"]
        cf_idx    = item["cf_index"]
        target_id = item["target_identity"]
        raw_text  = output.outputs[0].text if output.outputs else ""

        cf_text = clean_output(raw_text, row["text"])

        if cf_text and cf_text.strip().lower() == row["text"].strip().lower():
            cf_text = ""

        if not cf_text:
            fallback_count += 1
            cf_text = _injection_fallback(row["text"], target_id, cf_idx)

        results.append(_build_row(row["sample_id"], cf_text, row, cf_idx))

    log.info(
        "LLM inference done — %d generated  |  %d injection fallbacks (%.1f%%)",
        len(results), fallback_count, fallback_count / max(len(results), 1) * 100,
    )
    return results


llm_results = run_llm(llm_queue)

# %% [markdown]
# ## Cell 10 — Assemble, validate, and save

# %%
log.info("Assembling final dataset …")

original_rows = [
    {
        "original_sample_id": r["sample_id"],
        "counterfactual_id":  r["sample_id"],
        "text":               r["text"],
        "class_label":        r["class_label"],
        "target_group":       r["target_group"],
        "polarity":           r["polarity"],
        "hate_score":         r["hate_score"],
        "confidence":         r["confidence"],
        "cf_type":            "original",
        "t2i_prompt":         "",
    }
    for r in rows
]

# llm_results now contains all 12k counterfactuals (no phase1_results)
all_rows = original_rows + llm_results
df_final = pl.DataFrame(all_rows)

_cf_order = {"original": 0, "counterfactual_1": 1, "counterfactual_2": 2}
df_final = (
    df_final
    .with_columns(
        pl.col("cf_type").replace_strict(_cf_order, default=3).alias("_sort")
    )
    .sort(["original_sample_id", "_sort"])
    .drop("_sort")
)

# Integrity: every original must have exactly 3 rows
group_counts = df_final.group_by("original_sample_id").len()
orphans      = group_counts.filter(pl.col("len") != 3)
if len(orphans):
    log.warning("%d IDs without exactly 3 variants — dropping.", len(orphans))
    valid_ids = group_counts.filter(pl.col("len") == 3)["original_sample_id"]
    df_final  = df_final.filter(pl.col("original_sample_id").is_in(valid_ids))

df_final.write_csv(str(OUTPUT_CSV))

log.info("=" * 60)
log.info("SAVED  %d rows → %s", df_final.shape[0], OUTPUT_CSV)
log.info("cf_type:\n%s",       df_final["cf_type"].value_counts())
log.info("class_label:\n%s",   df_final["class_label"].value_counts())
log.info("=" * 60)

# %% [markdown]
# ## Cell 11 — Quality audit

# %%
print("=" * 70)
print("QUALITY AUDIT — 5 random samples per CF type")
print("=" * 70)

orig_lookup = {r["sample_id"]: r["text"] for r in rows}

for cf_type in ["counterfactual_1", "counterfactual_2"]:
    samples = df_final.filter(pl.col("cf_type") == cf_type).sample(5, seed=42).to_dicts()
    print(f"\n── {cf_type.upper()} ──")
    for r in samples:
        orig = orig_lookup.get(r["original_sample_id"], "?")
        print(f"\n  ID    : {r['original_sample_id']}")
        print(f"  CLASS : {r['class_label']}  |  TARGET: {r['target_group']}")
        print(f"  ORIG  : {orig[:110]}")
        print(f"  CF    : {r['text'][:110]}")

cf_only  = df_final.filter(pl.col("cf_type") != "original")
injected = cf_only.filter(
    pl.col("text").str.contains(r'\[about .+ people\]') |
    pl.col("text").str.contains(r'^\[.+\]:') |
    pl.col("text").str.contains(r'\(.+\)')
)
inj_pct = len(injected) / max(len(cf_only), 1) * 100
print(f"\n── Injection fallback rate: {len(injected)} / {len(cf_only)} ({inj_pct:.1f}%)")
if inj_pct > 15:
    print("⚠️  High fallback rate — consider lowering TEMPERATURE or increasing MAX_NEW_TOKENS.")
else:
    print("✅ Fallback rate within acceptable range.")


def _spot_check_identity_presence(df: pl.DataFrame, n: int = 50) -> float:
    sample = df.filter(pl.col("cf_type") != "original").sample(min(n, len(df)), seed=99)
    hits = 0
    for row in sample.to_dicts():
        if detect_identity_terms(row["text"]):
            hits += 1
    return hits / max(len(sample), 1)

id_hit_rate = _spot_check_identity_presence(df_final)
print(f"\n── Identity-term presence in CFs: {id_hit_rate:.1%}")
if id_hit_rate < 0.5:
    print("⚠️  Low identity-term presence — many CFs may be implicit-only. Review prompts.")
else:
    print("✅ Identity-term presence looks healthy.")

print(f"\n✅ Dataset saved → {OUTPUT_CSV}")