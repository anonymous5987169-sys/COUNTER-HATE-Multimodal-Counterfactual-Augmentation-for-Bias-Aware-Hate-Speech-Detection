"""
Build metadata.csv for CLIP scoring from the 18K dataset.

Maps each image file to its corresponding text and marks it as original or counterfactual.
This script creates the input CSV for compute_clip_scores.py.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Class label to image directory mapping
CLASS_TO_DIR = {
    "hate_race": "Hate/Hate_race/generated_images",
    "hate_religion": "Hate/Hate_religion/generated_images",
    "hate_gender": "Hate/Hate_gender/generated_images",
    "hate_other": "Hate/Hate_Others/generated_images",
    "offensive_non_hate": "non-hate/generated_images-offensive-non-hate",
    "neutral_discussion": "non-hate/generated_images-neutral",
    "counter_speech": "non-hate/generated_images-counter-speech",
    "ambiguous": "non-hate/generated_images-ambigious",
}


def extract_counterfactual_id_from_image_name(image_name):
    """
    Extract counterfactual_id from image file name.
    Examples: HS_HATERACE_0000.png -> HS_HATERACE_0000 (original)
              HS_HATERACE_0000_cf1.png -> HS_HATERACE_0000_cf1
    """
    return Path(image_name).stem


def build_metadata_csv(
    output_path="metadata.csv",
    dataset_csv="data/datasets/final_dataset_18k.csv",
    limit_per_class=None,
):
    """
    Build metadata.csv that maps images to texts and split types.
    
    Parameters
    ----------
    output_path : str
        Filename where to save metadata.csv (relative to results/)
    dataset_csv : str
        Path to the 18K dataset CSV (relative to project root, i.e., one level above src/)
    limit_per_class : int or None
        If set, limit to this many samples per class (for testing)
    """
    
    # Get paths
    clip_scoring_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(clip_scoring_dir, "results")
    src_dir = os.path.dirname(clip_scoring_dir)  # src/
    source_dir = os.path.dirname(src_dir)  # Source/
    project_root = os.path.dirname(source_dir)  # COUNTER-HATE root
    
    # Load the 18K dataset from project root
    csv_path = os.path.join(project_root, dataset_csv)
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df):,}")
    
    # Build a mapping: counterfactual_id -> (text, split_type, class_label)
    # split_type: "original" if cf_type == "original", else "counterfactual"
    mapping = {}
    for _, row in df.iterrows():
        cfid = row["counterfactual_id"]
        cf_type = row["cf_type"]
        split_type = "original" if cf_type == "original" else "counterfactual"
        mapping[cfid] = {
            "text": row["text"],
            "split_type": split_type,
            "class_label": row["class_label"],
        }
    
    print(f"  Built mapping for {len(mapping):,} counterfactual IDs")
    
    # Now scan image directories and match them to texts
    metadata_records = []
    
    for class_label, rel_dir in CLASS_TO_DIR.items():
        img_dir = os.path.join(project_root, rel_dir)
        if not os.path.isdir(img_dir):
            print(f"  WARNING: Image dir not found: {img_dir}")
            continue
        
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        print(f"  {class_label}: {len(image_files):,} images")
        
        # Limit per class if requested
        if limit_per_class is not None:
            image_files = image_files[:limit_per_class]
        
        for img_file in image_files:
            cfid = extract_counterfactual_id_from_image_name(img_file)
            
            if cfid not in mapping:
                print(f"    WARNING: No text mapping for image {img_file} (cfid={cfid})")
                continue
            
            info = mapping[cfid]
            img_path = os.path.join(rel_dir, img_file)  # relative path
            
            metadata_records.append({
                "image_path": img_path,
                "text": info["text"],
                "split_type": info["split_type"],
                "class_label": info["class_label"],
                "counterfactual_id": cfid,
            })
    
    metadata_df = pd.DataFrame(metadata_records)
    print(f"\nTotal metadata records: {len(metadata_df):,}")
    print(f"  Original: {(metadata_df['split_type'] == 'original').sum():,}")
    print(f"  Counterfactual: {(metadata_df['split_type'] == 'counterfactual').sum():,}")
    
    out_path = os.path.join(results_dir, output_path)
    os.makedirs(results_dir, exist_ok=True)
    metadata_df.to_csv(out_path, index=False)
    print(f"\nSaved metadata to {out_path}")
    
    return metadata_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build metadata.csv for CLIP scoring"
    )
    parser.add_argument(
        "--output", default="metadata.csv",
        help="Output CSV filename (relative to clip_scoring/)"
    )
    parser.add_argument(
        "--limit-per-class", type=int, default=None,
        help="Limit to N samples per class (for testing)"
    )
    args = parser.parse_args()
    
    build_metadata_csv(
        output_path=args.output,
        limit_per_class=args.limit_per_class,
    )
    print("Done!")
