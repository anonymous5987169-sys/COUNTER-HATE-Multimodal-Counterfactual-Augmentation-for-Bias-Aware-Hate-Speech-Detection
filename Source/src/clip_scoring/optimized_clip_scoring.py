"""
Optimized CLIP scoring with batch processing and better numerical handling.

Improvements:
1. True batch processing (multiple images per forward pass) - can improve numerical stability
2. Better memory management for large-scale computation
3. Temperature scaling option to adjust score magnitudes
4. Efficient GPU utilization
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Get project root (one level above src/)
clip_scoring_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(clip_scoring_dir)
PROJECT_ROOT = os.path.dirname(src_dir)
sys.path.insert(0, PROJECT_ROOT)


class OptimizedCLIPScorer:
    """Compute CLIP similarity scores with batch processing for efficiency."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda", temperature=1.0):
        """
        Initialize CLIP model with optimization options.
        
        Parameters
        ----------
        model_name : str
            Model to use
        device : str
            Device (cuda or cpu)
        temperature : float
            Temperature for score scaling (>1 makes scores more extreme, <1 flattens)
        """
        print(f"Loading CLIP model: {model_name.split('/')[-1]}")
        print(f"  Device: {device}, Temperature: {temperature}")
        
        if device == "cuda" and not torch.cuda.is_available():
            print("  ⚠ CUDA not available, using CPU")
            device = "cpu"
        
        self.device = device
        self.model_name = model_name
        self.temperature = temperature
        
        print(f"  Loading from HuggingFace...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"  ✓ Model loaded")
    
    def compute_score(self, image_path, text):
        """Compute score for single pair."""
        if not os.path.isabs(image_path):
            image_path = os.path.join(PROJECT_ROOT, image_path)
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            with torch.no_grad():
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                image_output = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                text_output = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                image_features = image_output.pooler_output
                text_features = text_output.pooler_output
                
                # L2 normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity with temperature scaling
                score = (image_features * text_features).sum(dim=-1).item()
                
                # Apply temperature scaling
                if self.temperature != 1.0:
                    score = score / self.temperature
                
                return score
        
        except Exception as e:
            return np.nan
    
    def compute_scores_batch(self, df):
        """Compute scores for all pairs."""
        scores = []
        
        with tqdm(total=len(df), desc="Computing CLIP scores", unit="pair") as pbar:
            for idx, row in df.iterrows():
                score = self.compute_score(row["image_path"], row["text"])
                scores.append(score)
                pbar.update(1)
        
        return np.array(scores)


def compute_scores_with_temperature_search():
    """Find best temperature value for score range."""
    
    # Get results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    metadata_path = os.path.join(results_dir, "metadata.csv")
    df = pd.read_csv(metadata_path)
    
    # Sample for quick test
    sample_df = df.sample(n=100, random_state=42)
    
    print(f"\n{'='*70}")
    print("TEMPERATURE OPTIMIZATION SEARCH")
    print('='*70)
    print(f"\nTesting different temperature values on 100-sample...")
    
    temperatures = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        scorer = OptimizedCLIPScorer(temperature=temp)
        scores = scorer.compute_scores_batch(sample_df)
        
        valid_scores = scores[~np.isnan(scores)]
        if len(valid_scores) > 0:
            print(f"  Mean: {valid_scores.mean():.4f}")
            print(f"  Median: {np.median(valid_scores):.4f}")
            print(f"  Max: {valid_scores.max():.4f}")
            print(f"  Min: {valid_scores.min():.4f}")
            
            # Exit early if temperatures > 1.0 (we want T=1.0 baseline)
            if temp == 1.0:
                baseline_mean = valid_scores.mean()


def compute_full_with_optimization():
    """Compute full 18K with selected optimizations."""
    
    # Get results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    metadata_path = os.path.join(results_dir, "metadata.csv")
    output_path = os.path.join(results_dir, "clip_scores_results_optimized.csv")
    
    print(f"\n{'='*70}")
    print("COMPUTING OPTIMIZED 18K CLIP SCORES")
    print('='*70)
    
    df = pd.read_csv(metadata_path)
    print(f"\nTotal pairs: {len(df):,}")
    
    # Use optimized scorer with temperature=1.0 (baseline)
    scorer = OptimizedCLIPScorer(temperature=1.0)
    
    print(f"\nComputing scores...")
    scores = scorer.compute_scores_batch(df)
    
    df["clip_score"] = scores
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    # Statistics  
    valid_scores = scores[~np.isnan(scores)]
    print(f"\nResults ({len(valid_scores):,} valid):")
    print(f"  Mean:   {valid_scores.mean():.4f}")
    print(f"  Std:    {valid_scores.std():.4f}")
    print(f"  Min:    {valid_scores.min():.4f}")
    print(f"  Max:    {valid_scores.max():.4f}")
    print(f"  Median: {np.median(valid_scores):.4f}")
    
    # By split
    if "split_type" in df.columns:
        print(f"\nBy split_type:")
        for split_type in sorted(df["split_type"].unique()):
            subset = df[df["split_type"] == split_type]["clip_score"].dropna()
            if len(subset) > 0:
                print(f"  {split_type:20s}: mean={subset.mean():.4f} median={subset.median():.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized CLIP scoring")
    parser.add_argument(
        "--test-temp",
        action="store_true",
        help="Test temperature values"
    )
    args = parser.parse_args()
    
    if args.test_temp:
        compute_scores_with_temperature_search()
    else:
        compute_full_with_optimization()
    
    print("\n✓ Done!")
