import argparse
import concurrent.futures
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset


# Keep HF logging minimal for long unattended runs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOD_ROOT = PROJECT_ROOT / "OOD-testing"
DATA_DIR = OOD_ROOT / "data"
RESULTS_DIR = OOD_ROOT / "results"
LOGS_DIR = OOD_ROOT / "logs"

TEXT_DATASET_ID = "dataspoof/HateXplain"
TEXT_SPLIT = "train"
IMAGE_DATASET_ID = "limjiayi/hateful_memes_expanded"
IMAGE_SPLITS = ["test_seen.jsonl", "test_unseen.jsonl"]

TEXT_MODEL_DIR = PROJECT_ROOT / "text_models" / "enhanced_results" / "models"
IMAGE_MODEL_DIR = PROJECT_ROOT / "image_models" / "models"

MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    nh = (y_true == 0)
    h = (y_true == 1)

    fpr = float((y_pred[nh] == 1).sum() / max(int(nh.sum()), 1))
    fnr = float((y_pred[h] == 0).sum() / max(int(h.sum()), 1))

    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "threshold": float(threshold),
    }


def load_text_ood_dataset() -> pd.DataFrame:
    ds = load_dataset(TEXT_DATASET_ID, split=TEXT_SPLIT)
    df = ds.to_pandas()

    # HateXplain labels in this mirror: normal/offensive/hatespeech.
    # We evaluate hate-speech detection with hatespeech=1, else=0.
    label_map = {
        "hatespeech": 1,
        "offensive": 0,
        "normal": 0,
    }

    df["raw_label"] = df["label"].astype(str).str.lower().str.strip()
    df["label_binary"] = df["raw_label"].map(label_map)
    df = df.dropna(subset=["text", "label_binary"]).copy()
    df["label_binary"] = df["label_binary"].astype(int)
    df = df.reset_index(drop=True)
    df["sample_id"] = [f"hatexplain_{i}" for i in range(len(df))]

    out_csv = DATA_DIR / "hatexplain_text_ood.csv"
    df[["sample_id", "text", "raw_label", "label_binary"]].to_csv(out_csv, index=False)
    return df[["sample_id", "text", "raw_label", "label_binary"]]


def evaluate_text_models(text_df: pd.DataFrame, batch_size: int) -> Dict[str, Dict]:
    encoder = SentenceTransformer(MINILM_MODEL_NAME)
    texts = text_df["text"].fillna("").tolist()
    y_true = text_df["label_binary"].to_numpy(dtype=int)

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = {}
    for cond in ["ncf", "cf"]:
        model_path = TEXT_MODEL_DIR / f"minilm_mlp_{cond}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing text checkpoint: {model_path}")

        clf = joblib.load(model_path)
        y_prob = clf.predict_proba(embeddings)[:, 1]
        metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)

        y_pred = (y_prob >= 0.5).astype(int)
        pred_df = text_df.copy()
        pred_df["pred_prob"] = y_prob
        pred_df["pred_label"] = y_pred
        pred_path = RESULTS_DIR / f"text_ood_predictions_{cond}.csv"
        pred_df.to_csv(pred_path, index=False)

        results[cond] = {
            "model": f"minilm_mlp_{cond}",
            "checkpoint": str(model_path.relative_to(PROJECT_ROOT)),
            "n_samples": int(len(y_true)),
            "metrics": metrics,
            "predictions_csv": str(pred_path.relative_to(PROJECT_ROOT)),
            "label_distribution": {
                "positive": int(y_true.sum()),
                "negative": int((1 - y_true).sum()),
            },
        }

    return results


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare_hateful_memes_test() -> pd.DataFrame:
    meta_paths = []
    for filename in IMAGE_SPLITS:
        local = hf_hub_download(
            repo_id=IMAGE_DATASET_ID,
            repo_type="dataset",
            filename=filename,
        )
        meta_paths.append(Path(local))

    rows: List[dict] = []
    for p in meta_paths:
        rows.extend(_read_jsonl(p))

    df = pd.DataFrame(rows)
    if not {"id", "img", "label"}.issubset(df.columns):
        raise ValueError("Unexpected Hateful Memes schema. Expected columns: id, img, label")

    images_dir = DATA_DIR / "hateful_memes_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rel_paths = df["img"].astype(str).tolist()
    local_paths = [""] * len(rel_paths)
    total = len(rel_paths)
    print(f"Downloading {total} Hateful Memes test images (parallel)...")

    def _download_one(item):
        idx, rel_path = item
        local = hf_hub_download(
            repo_id=IMAGE_DATASET_ID,
            repo_type="dataset",
            filename=rel_path,
            local_dir=str(images_dir),
            local_dir_use_symlinks=False,
        )
        return idx, local

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_download_one, pair) for pair in enumerate(rel_paths)]
        for fut in concurrent.futures.as_completed(futures):
            idx, local = fut.result()
            local_paths[idx] = local
            done += 1
            if done % 250 == 0 or done == total:
                print(f"  Downloaded {done}/{total} images")

    out = pd.DataFrame(
        {
            "sample_id": df["id"].astype(str).tolist(),
            "image_path": local_paths,
            "label_binary": df["label"].astype(int).tolist(),
            "meme_text": df.get("text", pd.Series([""] * len(df))).astype(str).tolist(),
            "hf_img_relpath": df["img"].astype(str).tolist(),
        }
    )

    out_csv = DATA_DIR / "hateful_memes_test_ood.csv"
    out.to_csv(out_csv, index=False)
    return out


class LocalImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image_path"]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return {
            "image": tensor,
            "label": int(row["label_binary"]),
            "sample_id": str(row["sample_id"]),
            "meme_text": str(row["meme_text"]),
            "hf_img_relpath": str(row["hf_img_relpath"]),
        }


def evaluate_image_models(img_df: pd.DataFrame, batch_size: int, num_workers: int) -> Dict[str, Dict]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from image_models.data_prep import eval_transforms
    from image_models.model import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LocalImageDataset(img_df, transform=eval_transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    conditions = {
        "ncf": "efficientnet_ncf.pth",
        "cf_no_adv": "efficientnet_cf_no_adv.pth",
        "cf": "efficientnet_cf.pth",
    }

    results: Dict[str, Dict] = {}
    y_true = img_df["label_binary"].to_numpy(dtype=int)

    for cond, ckpt_name in conditions.items():
        ckpt_path = IMAGE_MODEL_DIR / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing image checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        use_adversarial = bool(ckpt.get("use_adversarial", cond == "cf"))

        model = create_model(
            use_adversarial=use_adversarial,
            n_groups=8,
            dropout=cfg.get("dropout", 0.3),
            freeze_blocks=cfg.get("freeze_blocks", 6),
            device=device,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs: List[np.ndarray] = []
        sample_ids: List[str] = []
        meme_texts: List[str] = []
        relpaths: List[str] = []

        with torch.no_grad():
            for i, batch in enumerate(loader, start=1):
                logits, _ = model(batch["image"].to(device))
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(prob)
                sample_ids.extend(batch["sample_id"])
                meme_texts.extend(batch["meme_text"])
                relpaths.extend(batch["hf_img_relpath"])
                if i % 20 == 0:
                    print(f"  {cond}: processed {i * batch_size}/{len(dataset)} samples")

        y_prob = np.concatenate(probs, axis=0)
        metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)

        y_pred = (y_prob >= 0.5).astype(int)
        pred_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "hf_img_relpath": relpaths,
                "meme_text": meme_texts,
                "true_label": y_true,
                "pred_prob": y_prob,
                "pred_label": y_pred,
            }
        )
        pred_path = RESULTS_DIR / f"image_ood_predictions_{cond}.csv"
        pred_df.to_csv(pred_path, index=False)

        results[cond] = {
            "model": f"efficientnet_{cond}",
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
            "n_samples": int(len(y_true)),
            "metrics": metrics,
            "predictions_csv": str(pred_path.relative_to(PROJECT_ROOT)),
            "label_distribution": {
                "positive": int(y_true.sum()),
                "negative": int((1 - y_true).sum()),
            },
        }

    return results


def load_baselines() -> Dict[str, Dict[str, Dict[str, float]]]:
    image_path = PROJECT_ROOT / "image_models" / "results" / "evaluation_results.json"

    # Latest text baselines are taken from prof-report.md (MiniLM+MLP table):
    # nCF: F1=0.863, AUC=0.919, FPR=0.237
    # CF : F1=0.956, AUC=0.979, FPR=0.059
    # The corresponding latest baseline FNR values are not explicitly reported there.
    text_baseline = {
        "ncf": {
            "f1": 0.8630,
            "auc_roc": 0.9190,
            "fpr": 0.2370,
            "fnr": float("nan"),
        },
        "cf": {
            "f1": 0.9560,
            "auc_roc": 0.9790,
            "fpr": 0.0590,
            "fnr": float("nan"),
        },
    }

    img = json.loads(image_path.read_text(encoding="utf-8"))
    image_baseline = {}
    for cond in ["ncf", "cf_no_adv", "cf"]:
        m = img[cond]["metrics"]
        image_baseline[cond] = {
            "f1": float(m["f1"]),
            "auc_roc": float(m["auc_roc"]),
            "fpr": float(m["fpr"]),
            "fnr": float(m["fnr"]),
        }

    return {
        "text": text_baseline,
        "image": image_baseline,
    }


def _fmt(v: float) -> str:
    if v != v:  # NaN
        return "NA"
    return f"{v:.4f}"


def _pct(delta: float, base: float) -> str:
    if base == 0 or base != base:
        return "NA"
    return f"{(delta / base) * 100:.2f}%"


def write_results_md(
    text_results: Dict[str, Dict],
    image_results: Dict[str, Dict],
    baselines: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
) -> None:
    lines: List[str] = []

    lines.append("# OOD Testing Results: Text and Image Models")
    lines.append("")
    lines.append("## 1. Experimental Setup")
    lines.append("")
    lines.append("- Text OOD dataset (Hugging Face): `dataspoof/HateXplain` (split: train)")
    lines.append("- Image OOD dataset (Hugging Face): `limjiayi/hateful_memes_expanded` (splits: test_seen + test_unseen)")
    lines.append("- Text models tested: MiniLM+MLP `nCF`, `CF` checkpoints")
    lines.append("- Image models tested: EfficientNet `nCF`, `CF-no-adv`, `CF+GRL` checkpoints")
    lines.append("- Metrics reported: F1, AUC-ROC, FPR, FNR (primary OOD metrics)")
    lines.append("- Decision threshold: fixed `0.50` for OOD runs to keep condition comparisons consistent")
    lines.append("")
    lines.append("### Label Mapping Notes")
    lines.append("")
    lines.append("- HateXplain mapping used here: `hatespeech -> 1`, `offensive/normal -> 0`.")
    lines.append("- Hateful Memes labels are already binary (`1=hateful`, `0=non-hateful`).")
    lines.append("- OOD fairness is reported at aggregate level only (overall FPR/FNR), as requested.")
    lines.append("")

    lines.append("## 2. OOD Metrics (Direct Evaluation)")
    lines.append("")
    lines.append("### 2.1 Text Models on HateXplain")
    lines.append("")
    lines.append("| Condition | F1 | AUC | FPR | FNR | N |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in ["ncf", "cf"]:
        m = text_results[cond]["metrics"]
        n = text_results[cond]["n_samples"]
        lines.append(
            f"| {cond.upper()} | {_fmt(m['f1'])} | {_fmt(m['auc_roc'])} | {_fmt(m['fpr'])} | {_fmt(m['fnr'])} | {n} |"
        )
    lines.append("")

    lines.append("### 2.2 Image Models on Hateful Memes")
    lines.append("")
    lines.append("| Condition | F1 | AUC | FPR | FNR | N |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in ["ncf", "cf_no_adv", "cf"]:
        m = image_results[cond]["metrics"]
        n = image_results[cond]["n_samples"]
        lines.append(
            f"| {cond} | {_fmt(m['f1'])} | {_fmt(m['auc_roc'])} | {_fmt(m['fpr'])} | {_fmt(m['fnr'])} | {n} |"
        )
    lines.append("")

    lines.append("## 3. OOD vs Latest In-Distribution Baselines")
    lines.append("")
    lines.append("### 3.1 Text (MiniLM+MLP): OOD vs nCF/CF Latest")
    lines.append("")
    lines.append("Baseline source: `prof-report.md` (latest MiniLM+MLP table values)")
    lines.append("Note: latest text baseline FNR is not explicitly reported in `prof-report.md`; shown as `NA` below.")
    lines.append("")
    lines.append("| Condition | Metric | ID Baseline | OOD | Delta (OOD-ID) | Relative Change |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in ["ncf", "cf"]:
        b = baselines["text"][cond]
        o = text_results[cond]["metrics"]
        for key, label in [("f1", "F1"), ("auc_roc", "AUC"), ("fpr", "FPR"), ("fnr", "FNR")]:
            delta = o[key] - b[key]
            lines.append(
                f"| {cond.upper()} | {label} | {_fmt(b[key])} | {_fmt(o[key])} | {_fmt(delta)} | {_pct(delta, b[key])} |"
            )
    lines.append("")

    lines.append("### 3.2 Image (EfficientNet): OOD vs nCF/CF-no-adv/CF+GRL Latest")
    lines.append("")
    lines.append("Baseline source: `image_models/results/evaluation_results.json` (`metrics` block)")
    lines.append("")
    lines.append("| Condition | Metric | ID Baseline | OOD | Delta (OOD-ID) | Relative Change |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in ["ncf", "cf_no_adv", "cf"]:
        b = baselines["image"][cond]
        o = image_results[cond]["metrics"]
        for key, label in [("f1", "F1"), ("auc_roc", "AUC"), ("fpr", "FPR"), ("fnr", "FNR")]:
            delta = o[key] - b[key]
            lines.append(
                f"| {cond} | {label} | {_fmt(b[key])} | {_fmt(o[key])} | {_fmt(delta)} | {_pct(delta, b[key])} |"
            )
    lines.append("")

    lines.append("## 4. Findings")
    lines.append("")
    lines.append("1. This run measures **pure OOD generalization**: no retraining/fine-tuning was applied.")
    lines.append("2. Text OOD uses HateXplain from a public no-auth mirror and reports aggregate hate-vs-non-hate metrics.")
    lines.append("3. Image OOD uses Hateful Memes (test_seen + test_unseen) and evaluates image checkpoints only.")
    lines.append("4. Hateful Memes is inherently multimodal; this image-only evaluation intentionally ignores meme text and may understate fully multimodal performance.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OOD evaluation for canonical text and image models")
    parser.add_argument("--batch-size-text", type=int, default=128)
    parser.add_argument("--batch-size-image", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    run_manifest = {
        "text_dataset": {"id": TEXT_DATASET_ID, "split": TEXT_SPLIT},
        "image_dataset": {"id": IMAGE_DATASET_ID, "splits": IMAGE_SPLITS},
        "text_models": ["minilm_mlp_ncf.joblib", "minilm_mlp_cf.joblib"],
        "image_models": ["efficientnet_ncf.pth", "efficientnet_cf_no_adv.pth", "efficientnet_cf.pth"],
        "seed": args.seed,
        "batch_size_text": args.batch_size_text,
        "batch_size_image": args.batch_size_image,
        "num_workers": args.num_workers,
    }
    (RESULTS_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    print("[1/5] Loading OOD text dataset...")
    text_df = load_text_ood_dataset()

    print("[2/5] Running OOD text inference...")
    text_results = evaluate_text_models(text_df=text_df, batch_size=args.batch_size_text)

    print("[3/5] Preparing OOD image dataset...")
    image_df = prepare_hateful_memes_test()

    print("[4/5] Running OOD image inference...")
    image_results = evaluate_image_models(
        img_df=image_df,
        batch_size=args.batch_size_image,
        num_workers=args.num_workers,
    )

    print("[5/5] Building summary and markdown report...")
    baselines = load_baselines()

    summary = {
        "text_ood": text_results,
        "image_ood": image_results,
        "id_baselines": baselines,
    }
    (RESULTS_DIR / "ood_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_results_md(
        text_results=text_results,
        image_results=image_results,
        baselines=baselines,
        out_path=RESULTS_DIR / "results.md",
    )

    print("Done. Outputs:")
    print(f"- {RESULTS_DIR / 'ood_metrics_summary.json'}")
    print(f"- {RESULTS_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
