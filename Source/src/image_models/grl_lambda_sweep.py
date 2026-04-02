"""Run GRL adversarial-weight sweep for image models on CF condition.

This script trains CF+GRL models across a lambda grid and reports
validation-set metrics for both EfficientNet-B0 and CLIP ViT-B/32.

Output files:
  - image_models/results/grl_lambda_sweep_validation.json
  - image_models/results/grl_lambda_sweep_validation.csv
  - image_models/results/grl_lambda_sweep_validation.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Local imports
from data_prep import OUTPUT_DIR, create_dataloaders, get_condition_data, load_and_prepare
from evaluate import (
    compute_binary_metrics,
    compute_fairness_metrics,
    optimise_threshold,
    predict,
)
import train as train_mod
from train import DEFAULT_CONFIG, train_model

DEFAULT_LAMBDAS = [0.1, 0.3, 0.5, 0.7, 1.0]
DEFAULT_ARCHS = ["efficientnet_b0", "clip_vit_b32"]


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        raise ValueError("No lambda values provided.")
    return vals


def parse_str_list(raw: str) -> list[str]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("No architecture values provided.")
    return vals


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def subset_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).copy().reset_index(drop=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def write_markdown_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    headers = [
        "architecture",
        "adv_weight",
        "threshold",
        "val_f1",
        "val_auc",
        "val_fpr",
        "val_fnr",
        "val_eo_diff",
        "val_dp_diff",
        "train_time_s",
        "best_val_f1",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRL lambda sweep for image models")
    parser.add_argument(
        "--lambdas",
        type=str,
        default=",".join(str(v) for v in DEFAULT_LAMBDAS),
        help="Comma-separated lambda list, e.g. 0.1,0.3,0.5,0.7,1.0",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        default=",".join(DEFAULT_ARCHS),
        help="Comma-separated architecture list",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="Set -1 for auto",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lambdas = parse_float_list(args.lambdas)
    architectures = parse_str_list(args.architectures)
    device = resolve_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("GRL LAMBDA SWEEP (validation-set reporting)")
    print(f"Architectures: {architectures}")
    print(f"Lambdas      : {lambdas}")
    print(f"Device       : {device}")
    print("=" * 80)

    prepared = load_and_prepare()
    cf_data = get_condition_data(prepared, "cf")

    # Optional low-cost truncation knobs.
    cf_data["train"] = subset_df(cf_data["train"], args.max_train, args.seed)
    cf_data["val"] = subset_df(cf_data["val"], args.max_val, args.seed)

    results_dir = Path(OUTPUT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = results_dir / "lambda_sweep_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Keep sweep checkpoints separate from canonical pipeline checkpoints.
    train_mod.MODELS_DIR = str(model_dir)

    all_rows: list[dict[str, Any]] = []
    run_started = time.time()

    for arch in architectures:
        print("\n" + "-" * 80)
        print(f"Architecture: {arch}")
        print("-" * 80)

        num_workers = None if args.num_workers < 0 else args.num_workers
        loaders = create_dataloaders(
            cf_data,
            batch_size=args.batch_size,
            num_workers=num_workers,
            architecture=arch,
        )

        for lam in lambdas:
            print(f"\n[RUN] arch={arch} lambda={lam}")
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(
                {
                    "architecture": arch,
                    "adv_weight": float(lam),
                    "epochs": int(args.epochs),
                    "patience": int(args.patience),
                    "batch_size": int(args.batch_size),
                    "num_workers": 0 if num_workers is None else int(num_workers),
                }
            )

            t0 = time.time()
            train_out = train_model(condition="cf", loaders=loaders, config=cfg, device=device)
            train_time_s = time.time() - t0

            model = train_out["model"]
            y_val, y_val_prob, g_val = predict(model, loaders["val"], device=device)

            threshold, _ = optimise_threshold(y_val, y_val_prob)
            y_val_pred = (y_val_prob >= threshold).astype(int)
            y_val_pred_t050 = (y_val_prob >= 0.5).astype(int)

            m_val = compute_binary_metrics(y_val, y_val_pred, y_val_prob)
            m_val_t050 = compute_binary_metrics(y_val, y_val_pred_t050, y_val_prob)
            f_val = compute_fairness_metrics(y_val, y_val_pred, g_val)

            row = {
                "architecture": arch,
                "adv_weight": float(lam),
                "threshold": float(threshold),
                "val_f1": float(m_val["f1"]),
                "val_auc": float(m_val["auc_roc"]),
                "val_fpr": float(m_val["fpr"]),
                "val_fnr": float(m_val["fnr"]),
                "val_f1_t050": float(m_val_t050["f1"]),
                "val_fpr_t050": float(m_val_t050["fpr"]),
                "val_eo_diff": None
                if f_val["equalized_odds_diff"] is None
                else float(f_val["equalized_odds_diff"]),
                "val_dp_diff": None
                if f_val["demographic_parity_diff"] is None
                else float(f_val["demographic_parity_diff"]),
                "train_time_s": float(train_time_s),
                "best_val_f1": float(train_out.get("best_val_f1") or 0.0),
                "epochs": int(args.epochs),
                "patience": int(args.patience),
            }
            all_rows.append(row)

            print(
                "  val_f1={:.4f} val_auc={:.4f} val_fpr={:.4f} eo_diff={} train_s={:.1f}".format(
                    row["val_f1"],
                    row["val_auc"],
                    row["val_fpr"],
                    "None" if row["val_eo_diff"] is None else f"{row['val_eo_diff']:.4f}",
                    row["train_time_s"],
                )
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    all_rows.sort(key=lambda r: (r["architecture"], r["adv_weight"]))

    out_json = results_dir / "grl_lambda_sweep_validation.json"
    out_csv = results_dir / "grl_lambda_sweep_validation.csv"
    out_md = results_dir / "grl_lambda_sweep_validation.md"

    payload = {
        "description": "Validation-set GRL lambda sweep for image models",
        "architectures": architectures,
        "lambdas": lambdas,
        "condition": "cf",
        "device": str(device),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "max_train": int(args.max_train),
        "max_val": int(args.max_val),
        "runtime_seconds": round(time.time() - run_started, 2),
        "rows": all_rows,
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    write_markdown_table(all_rows, out_md)

    print("\n" + "=" * 80)
    print("Sweep complete")
    print(f"JSON: {out_json}")
    print(f"CSV : {out_csv}")
    print(f"MD  : {out_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
