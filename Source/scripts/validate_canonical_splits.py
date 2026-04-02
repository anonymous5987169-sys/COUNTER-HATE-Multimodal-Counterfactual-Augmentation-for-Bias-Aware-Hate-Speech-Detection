#!/usr/bin/env python3
"""Validate canonical split integrity.

Checks performed (core integrity):
1) Required keys/schema in canonical_splits.json
2) Pairwise disjointness of train/val/test IDs
3) Metadata counts match actual list lengths
4) Union coverage equals originals from source CSV after non-English filtering
5) Class distribution counts match IDs and metadata totals
6) Split ratio is close to 70/15/15 (integer-rounding tolerance)

Exit code:
  0 = VALID
  1 = INVALID
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "splits" / "canonical_splits.json"
DEFAULT_CSV = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k.csv"
FALLBACK_CSV = PROJECT_ROOT / "data" / "datasets" / "final_dataset_18k_t2i_prompts.csv"


def _is_non_english(text: str) -> bool:
    if not isinstance(text, str) or len(text) == 0:
        return False
    return sum(1 for char in text if ord(char) > 127) / len(text) > 0.05


def _find_csv(candidate: Path | None) -> Path:
    if candidate and candidate.exists():
        return candidate
    if DEFAULT_CSV.exists():
        return DEFAULT_CSV
    if FALLBACK_CSV.exists():
        return FALLBACK_CSV
    raise FileNotFoundError("Could not locate dataset CSV for validation")


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate(splits_path: Path, csv_path: Path | None = None) -> tuple[bool, list[str], dict[str, int]]:
    errors: list[str] = []

    if not splits_path.exists():
        return False, [f"Split file missing: {splits_path}"], {}

    with splits_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)

    required_top = {"metadata", "class_distribution", "train_ids", "val_ids", "test_ids"}
    missing = required_top - set(doc.keys())
    if missing:
        _error(errors, f"Missing top-level keys: {sorted(missing)}")
        return False, errors, {}

    for key in ["train_ids", "val_ids", "test_ids"]:
        if not isinstance(doc[key], list):
            _error(errors, f"{key} must be a list")

    train_ids = set(doc.get("train_ids", []))
    val_ids = set(doc.get("val_ids", []))
    test_ids = set(doc.get("test_ids", []))

    if len(train_ids) != len(doc.get("train_ids", [])):
        _error(errors, "Duplicate IDs found inside train_ids")
    if len(val_ids) != len(doc.get("val_ids", [])):
        _error(errors, "Duplicate IDs found inside val_ids")
    if len(test_ids) != len(doc.get("test_ids", [])):
        _error(errors, "Duplicate IDs found inside test_ids")

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv:
        _error(errors, f"Leakage train∩val: {len(overlap_tv)} IDs")
    if overlap_tt:
        _error(errors, f"Leakage train∩test: {len(overlap_tt)} IDs")
    if overlap_vt:
        _error(errors, f"Leakage val∩test: {len(overlap_vt)} IDs")

    meta = doc.get("metadata", {})
    expected_counts = {
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_total_originals": len(train_ids | val_ids | test_ids),
    }
    for key, observed in expected_counts.items():
        if key in meta and meta[key] != observed:
            _error(errors, f"Metadata mismatch for {key}: meta={meta[key]} actual={observed}")

    class_dist = doc.get("class_distribution", {})
    for split in ["train", "val", "test"]:
        if split not in class_dist or not isinstance(class_dist[split], dict):
            _error(errors, f"class_distribution.{split} missing or invalid")
            continue
        total_from_dist = sum(int(v) for v in class_dist[split].values())
        actual = len(doc[f"{split}_ids"])
        if total_from_dist != actual:
            _error(
                errors,
                f"class_distribution.{split} total={total_from_dist} != {split}_ids count={actual}",
            )

    csv_file = _find_csv(csv_path)
    df = pd.read_csv(csv_file)
    filtered = df[~df["text"].apply(_is_non_english)]
    originals = filtered[filtered["cf_type"] == "original"]
    original_ids = set(originals["original_sample_id"].astype(str).tolist())

    split_union = train_ids | val_ids | test_ids
    missing_from_splits = original_ids - split_union
    unknown_in_splits = split_union - original_ids
    if missing_from_splits:
        _error(errors, f"Original IDs missing from splits: {len(missing_from_splits)}")
    if unknown_in_splits:
        _error(errors, f"Split IDs not found in filtered originals: {len(unknown_in_splits)}")

    total = len(split_union)
    target_train = round(total * 0.70)
    target_val = round(total * 0.15)
    target_test = total - target_train - target_val
    tolerance = 3
    if abs(len(train_ids) - target_train) > tolerance:
        _error(errors, f"Train count off ratio target: got={len(train_ids)} target≈{target_train}")
    if abs(len(val_ids) - target_val) > tolerance:
        _error(errors, f"Val count off ratio target: got={len(val_ids)} target≈{target_val}")
    if abs(len(test_ids) - target_test) > tolerance:
        _error(errors, f"Test count off ratio target: got={len(test_ids)} target≈{target_test}")

    summary = {
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_total": total,
        "n_filtered_originals": len(original_ids),
    }
    return len(errors) == 0, errors, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical split integrity")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS, help="Path to canonical_splits.json")
    parser.add_argument("--csv", type=Path, default=None, help="Optional dataset CSV path")
    args = parser.parse_args()

    valid, errors, summary = validate(args.splits, args.csv)

    print("\n=== Canonical Split Validation ===")
    print(f"Splits file: {args.splits}")
    print(f"CSV source : {_find_csv(args.csv)}")
    if summary:
        print(
            "Counts     : "
            f"train={summary['n_train']} val={summary['n_val']} "
            f"test={summary['n_test']} total={summary['n_total']} "
            f"filtered_originals={summary['n_filtered_originals']}"
        )

    if valid:
        print("Result     : VALID ✅")
        return 0

    print("Result     : INVALID ❌")
    print("Issues:")
    for issue in errors:
        print(f"  - {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
