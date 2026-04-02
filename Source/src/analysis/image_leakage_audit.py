"""Audit image branch for leakage and shortcut artifacts.

Checks included:
1) Train/val/test overlap by original_sample_id and by image path.
2) Exact duplicate image content overlap using MD5 on original images.
3) Shortcut-risk probe: brightness gap between hate and non-hate image folders.

This script is read-only and does not modify dataset files.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ttest_ind


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "datasets", "final_dataset_18k.csv")
if not os.path.exists(DATA_CSV):
    DATA_CSV = os.path.join(PROJECT_ROOT, "data", "archive", "final_dataset_18k_t2i_prompts.csv")
SPLITS_JSON = os.path.join(PROJECT_ROOT, "data", "splits", "canonical_splits.json")

IMAGE_DIRS = {
    "hate_race": os.path.join(PROJECT_ROOT, "Hate", "Hate_race", "generated_images"),
    "hate_religion": os.path.join(PROJECT_ROOT, "Hate", "Hate_religion", "generated_images"),
    "hate_gender": os.path.join(PROJECT_ROOT, "Hate", "Hate_Gender", "generated_images"),
    "hate_other": os.path.join(PROJECT_ROOT, "Hate", "Hate_Others", "generated_images"),
    "ambiguous": os.path.join(PROJECT_ROOT, "non-hate", "generated_images-ambigious"),
    "counter_speech": os.path.join(PROJECT_ROOT, "non-hate", "generated_images-counter-speech"),
    "neutral_discussion": os.path.join(PROJECT_ROOT, "non-hate", "generated_images-neutral"),
    "offensive_non_hate": os.path.join(PROJECT_ROOT, "non-hate", "generated_images-offensive-non-hate"),
}


@dataclass
class OverlapReport:
    id_train_test: int
    id_train_val: int
    id_val_test: int
    path_train_test_ncf: int
    path_train_test_cf: int


@dataclass
class Md5Report:
    train_unique: int
    test_unique: int
    train_test_overlap: int
    duplicate_groups_all_originals: int


@dataclass
class ShortcutReport:
    sampled_hate: int
    sampled_non_hate: int
    gray_mean_hate: float
    gray_mean_non_hate: float
    gray_gap_non_minus_hate: float
    ttest_pvalue: float


def _load_df_with_image_paths() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    non_eng = df["text"].fillna("").apply(
        lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1) > 0.05
    )
    df = df.loc[~non_eng].copy()

    idx = {}
    for img_dir in IMAGE_DIRS.values():
        if not os.path.isdir(img_dir):
            continue
        for f in os.listdir(img_dir):
            if not f.endswith(".png"):
                continue
            path = os.path.join(img_dir, f)
            cfid = f[:-4]
            idx[cfid] = path
            idx[cfid.lower()] = path

    df["image_path"] = df["counterfactual_id"].astype(str).str.lower().map(idx)
    df = df.dropna(subset=["image_path"]).copy()
    return df


def _load_split_ids() -> tuple[set[str], set[str], set[str]]:
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    return set(splits["train_ids"]), set(splits["val_ids"]), set(splits["test_ids"])


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_overlap(df: pd.DataFrame, train_ids: set[str], val_ids: set[str], test_ids: set[str]) -> OverlapReport:
    originals = df[df["cf_type"] == "original"].copy()

    train_orig = originals[originals["original_sample_id"].isin(train_ids)]
    val_orig = originals[originals["original_sample_id"].isin(val_ids)]
    test_orig = originals[originals["original_sample_id"].isin(test_ids)]

    id_train = set(train_orig["original_sample_id"])
    id_val = set(val_orig["original_sample_id"])
    id_test = set(test_orig["original_sample_id"])

    ncf_train_paths = set(train_orig["image_path"])
    ncf_test_paths = set(test_orig["image_path"])

    cf_train = df[df["original_sample_id"].isin(train_ids)]
    cf_train_paths = set(cf_train["image_path"])

    return OverlapReport(
        id_train_test=len(id_train & id_test),
        id_train_val=len(id_train & id_val),
        id_val_test=len(id_val & id_test),
        path_train_test_ncf=len(ncf_train_paths & ncf_test_paths),
        path_train_test_cf=len(cf_train_paths & set(test_orig["image_path"])),
    )


def check_md5_content_overlap(df: pd.DataFrame, train_ids: set[str], test_ids: set[str]) -> Md5Report:
    originals = df[df["cf_type"] == "original"].copy()
    train = originals[originals["original_sample_id"].isin(train_ids)]
    test = originals[originals["original_sample_id"].isin(test_ids)]

    train_hashes = {_md5(p) for p in train["image_path"].unique()}
    test_hashes = {_md5(p) for p in test["image_path"].unique()}

    all_hashes = {}
    for p in originals["image_path"].unique():
        h = _md5(p)
        all_hashes[h] = all_hashes.get(h, 0) + 1

    dup_groups = sum(1 for c in all_hashes.values() if c > 1)

    return Md5Report(
        train_unique=len(train_hashes),
        test_unique=len(test_hashes),
        train_test_overlap=len(train_hashes & test_hashes),
        duplicate_groups_all_originals=dup_groups,
    )


def _sample_paths(paths: list[str], n: int) -> list[str]:
    if len(paths) <= n:
        return paths
    step = max(1, len(paths) // n)
    return paths[::step][:n]


def _gray_mean(path: str) -> float:
    arr = np.array(Image.open(path).convert("L").resize((32, 32)), dtype=np.float32) / 255.0
    return float(arr.mean())


def check_shortcut_risk(df: pd.DataFrame, sample_per_class: int = 120) -> ShortcutReport:
    hate_classes = {"hate_race", "hate_religion", "hate_gender", "hate_other"}
    group = (
        df[["class_label", "image_path"]]
        .drop_duplicates()
        .groupby("class_label")["image_path"]
        .apply(list)
        .to_dict()
    )

    hate_paths, non_paths = [], []
    for cls, paths in group.items():
        sampled = _sample_paths(paths, sample_per_class)
        if cls in hate_classes:
            hate_paths.extend(sampled)
        else:
            non_paths.extend(sampled)

    hate_vals = np.array([_gray_mean(p) for p in hate_paths], dtype=np.float32)
    non_vals = np.array([_gray_mean(p) for p in non_paths], dtype=np.float32)

    _stat, pvalue = ttest_ind(hate_vals, non_vals, equal_var=False)

    return ShortcutReport(
        sampled_hate=len(hate_vals),
        sampled_non_hate=len(non_vals),
        gray_mean_hate=float(hate_vals.mean()),
        gray_mean_non_hate=float(non_vals.mean()),
        gray_gap_non_minus_hate=float(non_vals.mean() - hate_vals.mean()),
        ttest_pvalue=float(pvalue),
    )


def main() -> None:
    print("=" * 72)
    print("IMAGE LEAKAGE / SHORTCUT AUDIT")
    print("=" * 72)

    df = _load_df_with_image_paths()
    train_ids, val_ids, test_ids = _load_split_ids()

    overlap = check_overlap(df, train_ids, val_ids, test_ids)
    md5 = check_md5_content_overlap(df, train_ids, test_ids)
    shortcut = check_shortcut_risk(df, sample_per_class=120)

    print("\n[1] Split overlap checks")
    print(f"  ID overlap train-test: {overlap.id_train_test}")
    print(f"  ID overlap train-val : {overlap.id_train_val}")
    print(f"  ID overlap val-test  : {overlap.id_val_test}")
    print(f"  Image-path overlap nCF train-test: {overlap.path_train_test_ncf}")
    print(f"  Image-path overlap CF  train-test: {overlap.path_train_test_cf}")

    print("\n[2] Exact content overlap checks (MD5 on originals)")
    print(f"  Unique train originals (by hash): {md5.train_unique}")
    print(f"  Unique test originals  (by hash): {md5.test_unique}")
    print(f"  Train-test hash overlap         : {md5.train_test_overlap}")
    print(f"  Duplicate-hash groups (all originals): {md5.duplicate_groups_all_originals}")

    print("\n[3] Shortcut-risk probe (brightness only)")
    print(f"  Sampled hate images    : {shortcut.sampled_hate}")
    print(f"  Sampled non-hate images: {shortcut.sampled_non_hate}")
    print(f"  Mean gray hate         : {shortcut.gray_mean_hate:.4f}")
    print(f"  Mean gray non-hate     : {shortcut.gray_mean_non_hate:.4f}")
    print(f"  Gap (non-hate - hate)  : {shortcut.gray_gap_non_minus_hate:.4f}")
    print(f"  Welch t-test p-value   : {shortcut.ttest_pvalue:.3e}")

    no_direct_leakage = (
        overlap.id_train_test == 0
        and overlap.id_train_val == 0
        and overlap.id_val_test == 0
        and overlap.path_train_test_ncf == 0
        and overlap.path_train_test_cf == 0
        and md5.train_test_overlap == 0
    )

    print("\n[Summary]")
    if no_direct_leakage:
        print("  No direct train/test leakage found in splits or exact image reuse.")
    else:
        print("  Potential direct leakage found. Investigate split/indexing logic.")

    if shortcut.ttest_pvalue < 1e-6 and abs(shortcut.gray_gap_non_minus_hate) > 0.03:
        print("  Strong class-conditioned visual artifact signal detected in generated images.")
        print("  Near-perfect image metrics are likely dominated by shortcut cues.")


if __name__ == "__main__":
    main()
