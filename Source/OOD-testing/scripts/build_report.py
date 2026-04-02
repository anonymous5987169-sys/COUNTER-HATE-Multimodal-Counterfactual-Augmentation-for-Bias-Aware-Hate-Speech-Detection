import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OOD_RESULTS = ROOT / "OOD-testing" / "results"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v):
    if v != v:
        return "NA"
    return f"{v:.4f}"


def pct(delta, base):
    if base == 0 or base != base:
        return "NA"
    return f"{(delta / base) * 100:.2f}%"


def main():
    text_ood = load_json(OOD_RESULTS / "text_ood_metrics.json")
    image_ood = load_json(OOD_RESULTS / "image_ood_metrics.json")

    comp = load_json(ROOT / "cross_modal" / "results" / "comprehensive_evaluation.json")
    rows = comp["results"]
    text_base = {}
    for cond in ["NCF", "CF"]:
        row = next(
            r
            for r in rows
            if r.get("modality") == "Text"
            and r.get("model") == "MiniLM + MLP"
            and r.get("condition") == cond
        )
        text_base[cond.lower()] = {
            "f1": row["macro_f1"],
            "auc_roc": row["auc_roc"],
            "fpr": row["fpr"],
            "fnr": row["fnr"],
        }

    img_base_all = load_json(ROOT / "image_models" / "results" / "evaluation_results.json")
    image_base = {
        cond: {
            "f1": img_base_all[cond]["metrics"]["f1"],
            "auc_roc": img_base_all[cond]["metrics"]["auc_roc"],
            "fpr": img_base_all[cond]["metrics"]["fpr"],
            "fnr": img_base_all[cond]["metrics"]["fnr"],
        }
        for cond in ["ncf", "cf_no_adv", "cf"]
    }

    lines = []
    lines.append("# OOD Testing Results")
    lines.append("")
    lines.append("## Datasets and Protocol")
    lines.append("")
    lines.append("- Text OOD dataset: dataspoof/HateXplain (train split, mapped to binary: hatespeech=1, offensive/normal=0)")
    lines.append("- Image OOD dataset: limjiayi/hateful_memes_expanded (test_seen + test_unseen)")
    lines.append("- Text models: MiniLM+MLP nCF and CF checkpoints")
    lines.append("- Image models: EfficientNet nCF, CF-no-adv, CF+GRL checkpoints")
    lines.append("- OOD metrics: F1, AUC-ROC, FPR, FNR at threshold 0.50")
    lines.append("- Fairness scope on OOD: aggregate only")
    lines.append("")

    lines.append("## OOD Metrics")
    lines.append("")
    lines.append("### Text OOD")
    lines.append("")
    lines.append("| Condition | F1 | AUC | FPR | FNR | N |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in ["ncf", "cf"]:
        m = text_ood[cond]["metrics"]
        n = text_ood[cond]["n_samples"]
        lines.append(f"| {cond.upper()} | {fmt(m['f1'])} | {fmt(m['auc_roc'])} | {fmt(m['fpr'])} | {fmt(m['fnr'])} | {n} |")
    lines.append("")

    lines.append("### Image OOD")
    lines.append("")
    lines.append("| Condition | F1 | AUC | FPR | FNR | N |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cond in ["ncf", "cf_no_adv", "cf"]:
        m = image_ood[cond]["metrics"]
        n = image_ood[cond]["n_samples"]
        lines.append(f"| {cond} | {fmt(m['f1'])} | {fmt(m['auc_roc'])} | {fmt(m['fpr'])} | {fmt(m['fnr'])} | {n} |")
    lines.append("")

    lines.append("## Comparison with Latest In-Distribution Results")
    lines.append("")
    lines.append("### Text (MiniLM+MLP): OOD vs ID")
    lines.append("")
    lines.append("| Condition | Metric | ID | OOD | Delta | Relative |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in ["ncf", "cf"]:
        for k, label in [("f1", "F1"), ("auc_roc", "AUC"), ("fpr", "FPR"), ("fnr", "FNR")]:
            b = text_base[cond][k]
            o = text_ood[cond]["metrics"][k]
            d = o - b
            lines.append(f"| {cond.upper()} | {label} | {fmt(b)} | {fmt(o)} | {fmt(d)} | {pct(d,b)} |")
    lines.append("")

    lines.append("### Image (EfficientNet): OOD vs ID")
    lines.append("")
    lines.append("| Condition | Metric | ID | OOD | Delta | Relative |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cond in ["ncf", "cf_no_adv", "cf"]:
        for k, label in [("f1", "F1"), ("auc_roc", "AUC"), ("fpr", "FPR"), ("fnr", "FNR")]:
            b = image_base[cond][k]
            o = image_ood[cond]["metrics"][k]
            d = o - b
            lines.append(f"| {cond} | {label} | {fmt(b)} | {fmt(o)} | {fmt(d)} | {pct(d,b)} |")
    lines.append("")

    lines.append("## Findings")
    lines.append("")
    lines.append("1. Text and image models were evaluated out-of-distribution with no retraining.")
    lines.append("2. Text OOD uses a binary mapping on HateXplain that is documented above.")
    lines.append("3. Image OOD uses meme images only; meme text was not fused during image-only inference.")
    lines.append("4. Aggregate FPR/FNR are reported as requested; per-group OOD fairness was not estimated.")
    lines.append("")

    out_path = OOD_RESULTS / "results.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
