"""
Validation script to verify all path fixes are working correctly.
"""

import os
import sys

# Get paths using the same logic as fixed scripts
# COUNTER-HATE root is 3 levels up from clip_scoring
clip_scoring_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(clip_scoring_dir)  # src/
source_dir = os.path.dirname(src_dir)  # Source/
project_root = os.path.dirname(source_dir)  # COUNTER-HATE root
results_dir = os.path.join(clip_scoring_dir, "results")

print("=" * 70)
print("CLIP-SCORING PATH VALIDATION")
print("=" * 70)

print("\nCalculated Paths:")
print(f"  clip_scoring_dir:  {clip_scoring_dir}")
print(f"  src_dir:           {src_dir}")
print(f"  source_dir:        {source_dir}")
print(f"  project_root:      {project_root}")
print(f"  results_dir:       {results_dir}")

print("\nDirectory Existence Checks:")
print(f"  ✓ clip_scoring exists:     {os.path.isdir(clip_scoring_dir)}")
print(f"  ✓ src/ exists:             {os.path.isdir(src_dir)}")
print(f"  ✓ Source/ exists:          {os.path.isdir(source_dir)}")
print(f"  ✓ project_root exists:     {os.path.isdir(project_root)}")
print(f"  ✓ results/ exists:         {os.path.isdir(results_dir)}")

print("\nDataset Path Checks:")
dataset_path = os.path.join(project_root, "data/datasets/final_dataset_18k.csv")
print(f"  Expected dataset path: {dataset_path}")
print(f"  Dataset exists:        {os.path.isfile(dataset_path)}")

print("\nImage Directory Checks:")
image_dirs = {
    "hate_race": "Hate/Hate_race/generated_images",
    "hate_religion": "Hate/Hate_religion/generated_images",
    "hate_gender": "Hate/Hate_gender/generated_images",
    "hate_other": "Hate/Hate_Others/generated_images",
    "offensive_non_hate": "non-hate/generated_images-offensive-non-hate",
    "neutral_discussion": "non-hate/generated_images-neutral",
    "counter_speech": "non-hate/generated_images-counter-speech",
    "ambiguous": "non-hate/generated_images-ambigious",
}

for label, rel_dir in image_dirs.items():
    img_dir = os.path.join(project_root, rel_dir)
    exists = os.path.isdir(img_dir)
    status = "✓" if exists else "✗"
    print(f"  {status} {label:30s}: {exists}")

print("\nResult Files Checks:")
result_files = [
    ("metadata.csv", os.path.join(results_dir, "metadata.csv")),
    ("clip_scores_results.csv", os.path.join(results_dir, "clip_scores_results.csv")),
    ("clip_summary_table.csv", os.path.join(results_dir, "clip_summary_table.csv")),
]

for fname, fpath in result_files:
    exists = os.path.isfile(fpath)
    status = "✓" if exists else "○"
    print(f"  {status} {fname:30s}: {exists}")

print("\nPath Construction Test (compute_clip_scores.py style):")
metadata_path = os.path.join(results_dir, "metadata.csv")
output_path = os.path.join(results_dir, "clip_scores_results.csv")
print(f"  Metadata input:  {metadata_path}")
print(f"  Scores output:   {output_path}")
print(f"  Forms directory: {results_dir}")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70 + "\n")
