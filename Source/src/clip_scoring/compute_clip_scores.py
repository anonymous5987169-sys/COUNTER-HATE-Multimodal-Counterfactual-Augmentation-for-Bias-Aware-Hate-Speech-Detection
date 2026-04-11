"""
Compute CLIP scores for (image, text) pairs.

Loads OpenAI CLIP model and computes text-image similarity scores for each pair.
Results are saved with an added 'clip_score' column and summary statistics printed.
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

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class CLIPScorer:
    """Compute CLIP similarity scores for image-text pairs."""
    
    def __init__(self, device="cuda", use_fp16=False):
        """
        Initialize CLIP model.
        
        Parameters
        ----------
        device : str
            Device to use ("cuda" or "cpu")
        use_fp16 : bool
            Whether to use float16 (generally not recommended for CLIP)
        """
        print(f"Loading CLIP model (device={device}, use_fp16={use_fp16})...")
        
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("  CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Load model and processor
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        if self.use_fp16:
            self.model = self.model.to(dtype)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        print(f"  Model loaded on {device} with dtype {dtype}")
    
    def compute_score(self, image_path, text, batch_size=1):
        """
        Compute CLIP score for a single (image, text) pair.
        
        Parameters
        ----------
        image_path : str
            Path to image file (absolute or relative to PROJECT_ROOT)
        text : str
            Text to compare with image
        batch_size : int
            Batch size (for consistency, though single pair is being processed)
        
        Returns
        -------
        float
            CLIP cosine similarity score between image and text features
        """
        # Ensure absolute path
        if not os.path.isabs(image_path):
            image_path = os.path.join(PROJECT_ROOT, image_path)
        
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Tokenize and process
            with torch.no_grad():
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                
                # Move to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                # Extract features - returns BaseModelOutputWithPooling objects
                image_output = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                text_output = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                # Extract the pooler output tensors (shape: [batch_size, 512])
                image_features = image_output.pooler_output
                text_features = text_output.pooler_output
                
                # L2 normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity as dot product of normalized features
                score = (image_features * text_features).sum(dim=-1).item()
            
            return score
        
        except Exception as e:
            print(f"  ERROR processing {image_path}: {e}")
            return np.nan
    
    def compute_scores_batch(self, df, batch_size=32):
        """
        Compute CLIP scores for all (image, text) pairs in a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must have columns 'image_path' and 'text'
        batch_size : int
            Batch size for processing
        
        Returns
        -------
        np.ndarray
            Array of scores, same length as df
        """
        scores = []
        
        with tqdm(total=len(df), desc="Computing CLIP scores", unit="pair") as pbar:
            for idx, row in df.iterrows():
                image_path = row["image_path"]
                text = row["text"]
                
                score = self.compute_score(image_path, text)
                scores.append(score)
                pbar.update(1)
        
        return np.array(scores)


def compute_clip_scores(
    metadata_csv="metadata.csv",
    output_csv="clip_scores_results.csv",
    batch_size=32,
    device="cuda",
    use_fp16=False,
):
    """
    Main function to compute CLIP scores.
    
    Parameters
    ----------
    metadata_csv : str
        Input metadata CSV (relative to clip_scoring/)
    output_csv : str
        Output CSV with clip_score column (relative to clip_scoring/)
    batch_size : int
        Batch size for processing
    device : str
        Device to use ("cuda" or "cpu")
    use_fp16 : bool
        Whether to use float16 precision (not recommended for CLIP)
    """
    
    metadata_path = os.path.join(PROJECT_ROOT, "clip_scoring", metadata_csv)
    output_path = os.path.join(PROJECT_ROOT, "clip_scoring", output_csv)
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    print(f"  Loaded {len(df):,} image-text pairs")
    print(f"  Columns: {list(df.columns)}")
    
    # Initialize CLIP scorer
    scorer = CLIPScorer(device=device, use_fp16=use_fp16)
    
    # Compute scores
    print(f"\nComputing CLIP scores (batch_size={batch_size})...")
    scores = scorer.compute_scores_batch(df, batch_size=batch_size)
    
    # Add scores to dataframe
    df["clip_score"] = scores
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute CLIP scores for image-text pairs")
    parser.add_argument("--metadata", default="metadata.csv", help="Input metadata CSV")
    parser.add_argument("--output", default="clip_scores_results.csv", help="Output CSV")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--use-fp16", action="store_true", help="Use float16 precision")
    
    args = parser.parse_args()
    
    compute_clip_scores(
        metadata_csv=args.metadata,
        output_csv=args.output,
        batch_size=args.batch_size,
        device=args.device,
        use_fp16=args.use_fp16,
    )
