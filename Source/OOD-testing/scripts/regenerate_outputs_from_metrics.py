import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OOD_RESULTS = PROJECT_ROOT / "OOD-testing" / "results"


def _fmt(v) -> str:
    if v is None:
        return "NA"
    if v != v:
        return "NA"
    return f"{v:.4f}"


def _pct(delta, base) -> str:
    if delta is None or base is None:
        return "NA"
    if base == 0 or base != base:
        return "NA"
    return f"{(delta / base) * 100:.2f}%"


def load_ood_metrics() -> Dict[str, Dict]:
    text_path = OOD_RESULTS / "text_ood_metrics.json"
    image_path = OOD_RESULTS / "image_ood_metrics.json"
    summary_path = OOD_RESULTS / "ood_metrics_summary.json"

    if not text_path.exists():
        raise FileNotFoundError(f"Missing {text_path}")
    text_ood = json.loads(text_path.read_text(encoding="utf-8"))

    if image_path.exists():
        image_ood = json.loads(image_path.read_text(encoding="utf-8"))
    elif summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        image_ood = existing.get("image_ood")
        if not image_ood:
            raise FileNotFoundError(
                f"Missing {image_path} and no image_ood block in {summary_path}"
            )
    else:
        raise FileNotFoundError(f"Missing {image_path} and {summary_path}")

    return {
        "text_ood": text_ood,
        "image_ood": image_ood,
    }


def load_latest_baselines() -> Dict[str, Dict[str, Dict[str, float]]]:
    image_eval_path = PROJECT_ROOT / "image_models" / "results" / "evaluation_results.json"
    image_eval = json.loads(image_eval_path.read_text(encoding="utf-8"))

    # Latest text baseline values from prof-report.md table.
    text_baseline = {
        "ncf": {"f1": 0.8630, "auc_roc": 0.9190, "fpr": 0.2370, "fnr": None},
        "cf": {"f1": 0.9560, "auc_roc": 0.9790, "fpr": 0.0590, "fnr": None},
    }

    image_baseline = {}
    for cond in ["ncf", "cf_no_adv", "cf"]:
        m = image_eval[cond]["metrics"]
        image_baseline[cond] = {
            "f1": float(m["f1"]),
            "auc_roc": float(m["auc_roc"]),
            "fpr": float(m["fpr"]),
            "fnr": float(m["fnr"]),
        }

    return {"text": text_baseline, "image": image_baseline}


def build_markdown(summary: Dict[str, Dict]) -> str:
    text_results = summary["text_ood"]
    image_results = summary["image_ood"]
    baselines = summary["id_baselines"]

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
            if b[key] is None or b[key] != b[key]:
                delta = None
            else:
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

    return "\n".join(lines) + "\n"


def main() -> None:
    summary = load_ood_metrics()
    summary["id_baselines"] = load_latest_baselines()

    out_json = OOD_RESULTS / "ood_metrics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_md = OOD_RESULTS / "results.md"
    out_md.write_text(build_markdown(summary), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
