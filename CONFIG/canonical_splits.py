"""
canonical_splits.py — Single source of truth for train/val/test splits.

All pipelines (text_models, image_models, cross_modal) must import and use
this module so that every model is evaluated on **identical** hold-out sets.

Strategy
────────
• Stratify on ``class_label`` (8-class) to preserve class proportions.
• Split at the *group* level: each ``original_sample_id`` appears in exactly
  one split.  Within CF conditions the train split is expanded to include all
  counterfactual variants, but val / test always contain originals only.
• Ratio target: 70 % train / 15 % val / 15 % test (of originals).
• Random state: 42 (fixed for reproducibility).

Persisted artefact
──────────────────
``data/splits/canonical_splits.json``  — saved automatically on first run
and reloaded on subsequent calls (unless ``force_recreate=True``).

Usage
─────
    # Standard import (add project root to sys.path first if needed)
    from canonical_splits import (
        get_canonical_splits,
        assign_split_column,
        build_condition_split_frames,
    )

    splits = get_canonical_splits()   # {train_ids, val_ids, test_ids} sets
    df["split"] = assign_split_column(df["original_sample_id"], splits)

    # Condition-aware split materialisation from a dataframe
    split_frames = build_condition_split_frames(
        df,
        condition="cf",
        augment_val_for_cf=True,
    )
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Paths ───────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _HERE  # canonical_splits.py lives at the project root

DATA_CSV = os.path.join(PROJECT_ROOT, "data", "datasets", "final_dataset_18k.csv")
if not os.path.exists(DATA_CSV):
    # Fall back to timestamped backup name used in some checkouts
    DATA_CSV = os.path.join(
        PROJECT_ROOT, "data", "datasets", "final_dataset_18k_t2i_prompts.csv"
    )

SPLITS_DIR  = os.path.join(PROJECT_ROOT, "data", "splits")
SPLITS_PATH = os.path.join(SPLITS_DIR, "canonical_splits.json")

# ─── Constants ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42

CLASS_LABELS: list[str] = [
    "hate_race", "hate_religion", "hate_gender", "hate_other",
    "offensive_non_hate", "neutral_discussion", "counter_speech", "ambiguous",
]

TARGET_GROUPS: list[str] = [
    "race/ethnicity", "religion", "gender", "sexual_orientation",
    "national_origin/citizenship", "disability", "age", "multiple/none",
]


# ═════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _remove_non_english(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows where >5 % of characters are non-ASCII (proxy for non-English)."""
    def _is_non_eng(text: str) -> bool:
        if not isinstance(text, str) or len(text) == 0:
            return False
        return sum(1 for c in text if ord(c) > 127) / len(text) > 0.05

    mask = df["text"].apply(_is_non_eng)
    n_removed = int(mask.sum())
    if n_removed:
        print(f"  [canonical_splits] Removing {n_removed} non-English rows "
              f"({n_removed / len(df) * 100:.2f} %)")
    return df[~mask].copy(), n_removed


def _build_splits(df_path: str | None = None) -> dict[str, Any]:
    """
    Build canonical splits from scratch.

    Reads the 18 k CSV, extracts originals, stratifies by class_label (8-class),
    performs a target 70 / 15 / 15 split on ``original_sample_id``.

    Returns a serialisable dict ready for JSON persistence.
    """
    csv_path = df_path or DATA_CSV
    print(f"\n[canonical_splits] Building splits from {os.path.basename(csv_path)} …")

    df = pd.read_csv(csv_path)
    print(f"  Full dataset: {len(df):,} rows")

    n_originals_raw = int((df["cf_type"] == "original").sum())

    # Filter non-English
    df, _ = _remove_non_english(df)

    # Extract originals only for group-level splitting
    originals = df[df["cf_type"] == "original"].copy()
    n_removed_non_english = n_originals_raw - len(originals)
    print(f"  Originals: {len(originals):,} rows")

    # ── Validate class labels ───────────────────────────────────────────
    unknown = set(originals["class_label"].unique()) - set(CLASS_LABELS)
    if unknown:
        print(f"  WARNING: unknown class labels found: {unknown}")

    # ── 70 / 15 / 15 stratified split ──────────────────────────────────
    train_orig, temp_orig = train_test_split(
        originals,
        test_size=0.30,
        stratify=originals["class_label"],
        random_state=RANDOM_STATE,
    )
    val_orig, test_orig = train_test_split(
        temp_orig,
        test_size=0.50,
        stratify=temp_orig["class_label"],
        random_state=RANDOM_STATE,
    )

    train_ids: list[str] = sorted(train_orig["original_sample_id"].tolist())
    val_ids:   list[str] = sorted(val_orig["original_sample_id"].tolist())
    test_ids:  list[str] = sorted(test_orig["original_sample_id"].tolist())

    # ── Sanity checks ───────────────────────────────────────────────────
    assert set(train_ids).isdisjoint(set(val_ids)),  "LEAKAGE: train ∩ val!"
    assert set(train_ids).isdisjoint(set(test_ids)), "LEAKAGE: train ∩ test!"
    assert set(val_ids).isdisjoint(set(test_ids)),   "LEAKAGE: val ∩ test!"
    print(f"  ✓ Splits are disjoint — train={len(train_ids):,} | "
          f"val={len(val_ids):,} | test={len(test_ids):,}")

    # ── Class distribution by split ─────────────────────────────────────
    def _class_dist(sub_df: pd.DataFrame) -> dict[str, int]:
        return sub_df["class_label"].value_counts().to_dict()

    realized_train = len(train_ids) / len(originals) * 100
    realized_val = len(val_ids) / len(originals) * 100
    realized_test = len(test_ids) / len(originals) * 100

    splits_doc = {
        "metadata": {
            "random_state": RANDOM_STATE,
            "n_train": len(train_ids),
            "n_val":   len(val_ids),
            "n_test":  len(test_ids),
            "n_originals_raw": n_originals_raw,
            "n_total_originals": len(originals),
            "n_removed_non_english": int(n_removed_non_english),
            "stratify_on": "class_label",
            "split_ratio": "70/15/15",
            "split_ratio_realized": (
                f"{realized_train:.2f}/{realized_val:.2f}/{realized_test:.2f}"
            ),
            "csv_source": os.path.basename(csv_path),
            "created_at": str(date.today()),
        },
        "class_distribution": {
            "train": _class_dist(train_orig),
            "val":   _class_dist(val_orig),
            "test":  _class_dist(test_orig),
        },
        "train_ids": train_ids,
        "val_ids":   val_ids,
        "test_ids":  test_ids,
    }
    return splits_doc


def _save_splits(splits_doc: dict[str, Any], path: str = SPLITS_PATH) -> None:
    """Persist splits document to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(splits_doc, fh, indent=2)
    print(f"  [canonical_splits] Saved → {path}")


def _load_splits(path: str = SPLITS_PATH) -> dict[str, Any]:
    """Load persisted splits document from JSON."""
    with open(path) as fh:
        return json.load(fh)


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

# Module-level cache so repeated imports don't trigger disk I/O
_SPLITS_CACHE: dict[str, Any] | None = None


def get_canonical_splits(
    force_recreate: bool = False,
    df_path: str | None = None,
    save_path: str = SPLITS_PATH,
) -> dict[str, set[str]]:
    """
    Return canonical train / val / test split ID sets.

    Loads from ``data/splits/canonical_splits.json`` when available.
    Creates and persists the file on first call.

    Parameters
    ----------
    force_recreate : bool
        If True, rebuild splits even if the JSON file already exists.
    df_path : str | None
        Override the default 18 k CSV path.
    save_path : str
        Where to save / load the canonical splits JSON.

    Returns
    -------
    dict with keys ``train_ids``, ``val_ids``, ``test_ids`` — each a *set* of
    ``original_sample_id`` strings.
    """
    global _SPLITS_CACHE

    if not force_recreate and _SPLITS_CACHE is not None:
        return _SPLITS_CACHE

    if not force_recreate and os.path.exists(save_path):
        print(f"[canonical_splits] Loading existing splits from "
              f"{os.path.basename(save_path)}")
        doc = _load_splits(save_path)
    else:
        doc = _build_splits(df_path)
        _save_splits(doc, save_path)

    result: dict[str, set[str]] = {
        "train_ids": set(doc["train_ids"]),
        "val_ids":   set(doc["val_ids"]),
        "test_ids":  set(doc["test_ids"]),
    }
    _SPLITS_CACHE = result
    return result


def assign_split_column(
    id_series: pd.Series,
    splits: dict[str, set[str]] | None = None,
) -> pd.Series:
    """
    Map a Series of ``original_sample_id`` values to split labels.

    Parameters
    ----------
    id_series : pd.Series
        Series of ``original_sample_id`` strings.
    splits : dict | None
        Result of ``get_canonical_splits()``.  Loaded automatically if None.

    Returns
    -------
    pd.Series of str with values ``'train'`` | ``'val'`` | ``'test'`` | ``'unknown'``.
    """
    if splits is None:
        splits = get_canonical_splits()

    def _label(sid: str) -> str:
        if sid in splits["train_ids"]:
            return "train"
        if sid in splits["val_ids"]:
            return "val"
        if sid in splits["test_ids"]:
            return "test"
        return "unknown"

    return id_series.map(_label)


def build_condition_split_frames(
    df: pd.DataFrame,
    condition: str,
    splits: dict[str, set[str]] | None = None,
    augment_val_for_cf: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Build train/val/test frames for a condition from canonical ID splits.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least ``original_sample_id`` and ``cf_type``.
        May contain originals only (nCF) or originals+counterfactuals (CF).
    condition : str
        ``'ncf'`` or ``'cf'`` (``'cf_no_adv'`` aliases to CF behaviour).
    splits : dict | None
        Canonical splits from ``get_canonical_splits()``; loaded if None.
    augment_val_for_cf : bool
        When True and ``condition`` is CF-like, validation includes all variants
        for val IDs (original + counterfactual rows). Test remains originals only.

    Returns
    -------
    dict with keys ``train``, ``val``, ``test``.
    """
    if "original_sample_id" not in df.columns or "cf_type" not in df.columns:
        raise ValueError(
            "build_condition_split_frames requires columns: "
            "'original_sample_id' and 'cf_type'."
        )

    if splits is None:
        splits = get_canonical_splits()

    cond = str(condition).strip().lower()
    if cond == "cf_no_adv":
        cond = "cf"
    if cond not in {"ncf", "cf"}:
        raise ValueError(f"Unsupported condition: {condition}")

    train_ids = splits["train_ids"]
    val_ids = splits["val_ids"]
    test_ids = splits["test_ids"]

    originals = df[df["cf_type"] == "original"].copy()
    train_orig = originals[originals["original_sample_id"].isin(train_ids)].copy()
    val_orig = originals[originals["original_sample_id"].isin(val_ids)].copy()
    test_orig = originals[originals["original_sample_id"].isin(test_ids)].copy()

    if cond == "ncf":
        return {"train": train_orig, "val": val_orig, "test": test_orig}

    train_df = df[df["original_sample_id"].isin(train_ids)].copy()
    if augment_val_for_cf:
        val_df = df[df["original_sample_id"].isin(val_ids)].copy()
    else:
        val_df = val_orig

    return {"train": train_df, "val": val_df, "test": test_orig}


# ═════════════════════════════════════════════════════════════════════════════
#  CLI — run standalone to (re)generate the JSON artefact
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate / verify canonical train/val/test splits."
    )
    parser.add_argument(
        "--force-recreate", action="store_true",
        help="Rebuild and overwrite existing splits JSON.",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Override path to the 18 k dataset CSV.",
    )
    args = parser.parse_args()

    splits = get_canonical_splits(
        force_recreate=args.force_recreate,
        df_path=args.csv,
    )

    print(f"\nCanonical split summary:")
    print(f"  train : {len(splits['train_ids']):,} original IDs")
    print(f"  val   : {len(splits['val_ids']):,} original IDs")
    print(f"  test  : {len(splits['test_ids']):,} original IDs")
    print(f"  total : {sum(len(v) for v in splits.values()):,} IDs")
    print(f"\nPersisted → {SPLITS_PATH}")
