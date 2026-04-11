"""
Quick test of fixed CLIP scoring on a small sample.
"""

import os
import sys
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Get paths
clip_scoring_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(clip_scoring_dir, "results")

# Load metadata
metadata_path = os.path.join(results_dir, "metadata.csv")
df = pd.read_csv(metadata_path)
print(f"Loaded {len(df):,} pairs from metadata")

# Take first 50 samples for quick test
df_test = df.head(50)

# Get project root
src_dir = os.path.dirname(clip_scoring_dir)
project_root = os.path.dirname(src_dir)

# Load CLIP model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("Model loaded\n")

# Compute scores on test set
scores = []
for idx, row in df_test.iterrows():
    image_path = row["image_path"]
    text = row["text"]
    
    # Ensure absolute path
    if not os.path.isabs(image_path):
        image_path = os.path.join(project_root, image_path)
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to("cuda")
            
            # FIXED: Extract features directly and compute cosine similarity
            image_output = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_output = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            # Extract pooler output tensors
            image_features = image_output.pooler_output
            text_features = text_output.pooler_output
            
            # L2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            score = (image_features * text_features).sum(dim=-1).item()
            scores.append(score)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/50")
    
    except Exception as e:
        print(f"  ERROR at {idx}: {e}")
        scores.append(None)

print(f"\nTest Results (50 samples):")
print(f"  Mean:   {pd.Series(scores).mean():.4f}")
print(f"  Std:    {pd.Series(scores).std():.4f}")
print(f"  Min:    {pd.Series(scores).min():.4f}")
print(f"  Max:    {pd.Series(scores).max():.4f}")
print(f"\nFirst 10 scores: {scores[:10]}")
