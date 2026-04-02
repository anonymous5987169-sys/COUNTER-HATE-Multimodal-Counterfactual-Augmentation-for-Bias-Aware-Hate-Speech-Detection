import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOD_ROOT = PROJECT_ROOT / "OOD-testing"
DATA_DIR = OOD_ROOT / "data"
RESULTS_DIR = OOD_ROOT / "results"

DATASET_ID = "limjiayi/hateful_memes_expanded"
SPLITS = ["test_seen.jsonl", "test_unseen.jsonl"]
IMAGE_MODEL_DIR = PROJECT_ROOT / "image_models" / "models"


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    nh = (y_true == 0)
    h = (y_true == 1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "fpr": float((y_pred[nh] == 1).sum() / max(int(nh.sum()), 1)),
        "fnr": float((y_pred[h] == 0).sum() / max(int(h.sum()), 1)),
        "threshold": float(threshold),
    }


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = DATA_DIR / "hateful_memes_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for filename in SPLITS:
        path = hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=filename)
        rows.extend(_read_jsonl(Path(path)))

    df = pd.DataFrame(rows)

    local_paths = []
    total = len(df)
    print(f"Downloading {total} test images...")
    for idx, rel in enumerate(df["img"].tolist(), start=1):
        local = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=str(rel),
            local_dir=str(images_dir),
            local_dir_use_symlinks=False,
        )
        local_paths.append(local)
        if idx % 250 == 0 or idx == total:
            print(f"  Downloaded {idx}/{total}")

    out = pd.DataFrame(
        {
            "sample_id": df["id"].astype(str),
            "image_path": local_paths,
            "label_binary": df["label"].astype(int),
            "meme_text": df.get("text", "").astype(str),
            "hf_img_relpath": df["img"].astype(str),
        }
    )
    out.to_csv(DATA_DIR / "hateful_memes_test_ood.csv", index=False)
    return out


class LocalImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        return {
            "image": self.transform(image),
            "label": int(row["label_binary"]),
            "sample_id": str(row["sample_id"]),
            "meme_text": str(row["meme_text"]),
            "hf_img_relpath": str(row["hf_img_relpath"]),
        }


def evaluate(df: pd.DataFrame, batch_size: int = 64) -> Dict[str, Dict]:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from image_models.data_prep import eval_transforms
    from image_models.model import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LocalImageDataset(df=df, transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true = df["label_binary"].to_numpy(dtype=int)

    conditions = {
        "ncf": "efficientnet_ncf.pth",
        "cf_no_adv": "efficientnet_cf_no_adv.pth",
        "cf": "efficientnet_cf.pth",
    }

    out = {}
    for cond, name in conditions.items():
        ckpt = torch.load(IMAGE_MODEL_DIR / name, map_location=device)
        cfg = ckpt.get("config", {})
        model = create_model(
            use_adversarial=bool(ckpt.get("use_adversarial", cond == "cf")),
            n_groups=8,
            dropout=cfg.get("dropout", 0.3),
            freeze_blocks=cfg.get("freeze_blocks", 6),
            device=device,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        probs = []
        sample_ids = []
        texts = []
        relpaths = []

        with torch.no_grad():
            for i, batch in enumerate(loader, start=1):
                logits, _ = model(batch["image"].to(device))
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)
                sample_ids.extend(batch["sample_id"])
                texts.extend(batch["meme_text"])
                relpaths.extend(batch["hf_img_relpath"])
                if i % 20 == 0:
                    print(f"  {cond}: batch {i}/{len(loader)}")

        y_prob = np.concatenate(probs)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

        pred_df = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "hf_img_relpath": relpaths,
                "meme_text": texts,
                "true_label": y_true,
                "pred_prob": y_prob,
                "pred_label": y_pred,
            }
        )
        pred_path = RESULTS_DIR / f"image_ood_predictions_{cond}.csv"
        pred_df.to_csv(pred_path, index=False)

        out[cond] = {
            "checkpoint": str((IMAGE_MODEL_DIR / name).relative_to(PROJECT_ROOT)),
            "n_samples": int(len(y_true)),
            "metrics": metrics,
            "predictions_csv": str(pred_path.relative_to(PROJECT_ROOT)),
        }

    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/2] Preparing image OOD dataset")
    df = prepare_dataset()

    print("[2/2] Running image model evaluation")
    res = evaluate(df)

    out_path = RESULTS_DIR / "image_ood_metrics.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
