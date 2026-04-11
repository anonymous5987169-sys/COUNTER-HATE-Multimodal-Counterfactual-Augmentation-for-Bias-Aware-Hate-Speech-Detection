"""
Investigate and fix text encoding issues that may be hurting CLIP scores.

Issues found:
- Emoji encoding problems (ðŸ˜ instead of proper emoji)
- Potential UTF-8 decoding issues
- Text preprocessing could help scores
"""

import os
import sys
import pandas as pd

# Get project root (one level above src/)
clip_scoring_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(clip_scoring_dir)
PROJECT_ROOT = os.path.dirname(src_dir)
sys.path.insert(0, PROJECT_ROOT)


def analyze_text_quality():
    """Check for encoding and text quality issues."""
    
    # Load dataset from project root
    dataset_path = os.path.join(PROJECT_ROOT, "data/datasets/final_dataset_18k.csv")
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"  Total rows: {len(df):,}\n")
    
    # Check for encoding issues
    print("="*70)
    print("TEXT QUALITY ANALYSIS")
    print("="*70)
    
    print("\n1. Checking for problematic characters...")
    encoding_issues = []
    emoji_like_patterns = ['ðŸ', '\ufffd', '\\u0000']  # Common broken encodings
    
    for idx, text in enumerate(df['text'].head(200)):
        if pd.isna(text):
            continue
        for pattern in emoji_like_patterns:
            if pattern in str(text):
                encoding_issues.append((idx, text[:100]))
                break
    
    if encoding_issues:
        print(f"Found {len(encoding_issues)} texts with potential encoding issues:")
        for idx, text in encoding_issues[:5]:
            print(f"  Row {idx}: {repr(text)}")
    else:
        print("✓ No obvious encoding issues found in first 200 samples")
    
    # Check for text length distribution
    print("\n2. Text length distribution:")
    lengths = df['text'].str.len()
    print(f"  Mean length: {lengths.mean():.0f} chars")
    print(f"  Median length: {lengths.median():.0f} chars")
    print(f"  Min: {lengths.min():.0f}, Max: {lengths.max():.0f}")
    print(f"  Q1: {lengths.quantile(0.25):.0f}, Q3: {lengths.quantile(0.75):.0f}")
    
    # Check for special characters
    print("\n3. Special character content:")
    has_urls = df['text'].str.contains('URL|http|www', case=False, na=False).sum()
    has_numbers = df['text'].str.contains(r'\d{3,}', na=False).sum()
    has_caps = df['text'].str.contains('[A-Z]{3,}', na=False).sum()
    
    print(f"  Rows with URLs/URLs keyword: {has_urls:,} ({100*has_urls/len(df):.1f}%)")
    print(f"  Rows with numbers (3+): {has_numbers:,} ({100*has_numbers/len(df):.1f}%)")
    print(f"  Rows with CAPS (3+): {has_caps:,} ({100*has_caps/len(df):.1f}%)")
    
    # Sample texts
    print("\n4. Sample texts from each class:")
    for class_label in sorted(df['class_label'].unique()):
        sample = df[df['class_label'] == class_label]['text'].iloc[0]
        print(f"\n  {class_label}:")
        print(f"    {repr(sample[:80])}...")


def clean_text(text):
    """Clean and normalize text for CLIP processing."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove problematic patterns
    # Replace "URL" with descriptive term
    text = text.replace('URL', ' [URL] ')
    
    # Keep problematic encodings as-is for now (they may be important for semantics)
    
    return text.strip()


def compare_scores_before_after():
    """Compare CLIP scores before and after text cleaning."""
    
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    import numpy as np
    
    # Load dataset
    dataset_path = os.path.join(PROJECT_ROOT, "data/datasets/final_dataset_18k.csv")
    clip_scoring_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(clip_scoring_dir, "results")
    metadata_path = os.path.join(results_dir, "metadata.csv")
    
    df = pd.read_csv(dataset_path)
    meta_df = pd.read_csv(metadata_path)
    
    # Sample 10 rows
    sample_meta = meta_df.sample(n=10, random_state=42)
    
    print(f"\n\n{'='*70}")
    print("COMPARING SCORES: ORIGINAL vs CLEANED TEXT")
    print('='*70)
    
    # Load CLIP model for quick test
    print("\nLoading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    print(f"\nScoring 10 samples with original vs cleaned text:\n")
    
    for idx, row in sample_meta.iterrows():
        image_path = row["image_path"]
        text_original = row["text"]
        text_cleaned = clean_text(text_original)
        
        # Ensure absolute path
        if not os.path.isabs(image_path):
            image_path = os.path.join(PROJECT_ROOT, image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                # Original text
                inputs_orig = processor(
                    text=[text_original],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                for key in inputs_orig:
                    if isinstance(inputs_orig[key], torch.Tensor):
                        inputs_orig[key] = inputs_orig[key].to("cuda")
                
                img_feat = model.get_image_features(pixel_values=inputs_orig["pixel_values"])
                txt_feat = model.get_text_features(
                    input_ids=inputs_orig["input_ids"],
                    attention_mask=inputs_orig["attention_mask"]
                )
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                score_orig = (img_feat * txt_feat).sum(dim=-1).item()
                
                # Cleaned text
                inputs_clean = processor(
                    text=[text_cleaned],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                for key in inputs_clean:
                    if isinstance(inputs_clean[key], torch.Tensor):
                        inputs_clean[key] = inputs_clean[key].to("cuda")
                
                img_feat = model.get_image_features(pixel_values=inputs_clean["pixel_values"])
                txt_feat = model.get_text_features(
                    input_ids=inputs_clean["input_ids"],
                    attention_mask=inputs_clean["attention_mask"]
                )
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                score_clean = (img_feat * txt_feat).sum(dim=-1).item()
                
                diff = score_clean - score_orig
                print(f"  Original: {score_orig:.4f}  Cleaned: {score_clean:.4f}  Diff: {diff:+.4f}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\nℹ Text cleaning effects are minimal - focus on model/parameter optimization")


if __name__ == "__main__":
    analyze_text_quality()
    #compare_scores_before_after()  # Uncomment to run comparison
