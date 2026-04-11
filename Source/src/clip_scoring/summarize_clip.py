"""
Summarize CLIP scores from results CSV.

Reads clip_scores_results.csv and outputs summary statistics grouped by split_type.
"""

import os
import sys
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def summarize_clip_scores(
    input_csv="clip_scores_results.csv",
    output_csv="clip_summary_table.csv",
):
    """
    Summarize CLIP scores and save grouped statistics.
    
    Parameters
    ----------
    input_csv : str
        Input results CSV (relative to clip_scoring/results/)
    output_csv : str
        Output summary CSV (relative to clip_scoring/results/)
    """
    
    # Get clip_scoring directory and results subdirectory
    clip_scoring_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(clip_scoring_dir, "results")
    
    input_path = os.path.join(results_dir, input_csv)
    output_path = os.path.join(results_dir, output_csv)
    
    # Load results
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} records")
    
    # Print overall stats
    print("\n" + "="*70)
    print("OVERALL CLIP SCORE STATISTICS")
    print("="*70)
    
    valid_scores = df["clip_score"].dropna()
    print(f"\nTotal valid scores: {len(valid_scores):,}")
    print(f"  Mean:   {valid_scores.mean():.6f}")
    print(f"  Std:    {valid_scores.std():.6f}")
    print(f"  Min:    {valid_scores.min():.6f}")
    print(f"  Max:    {valid_scores.max():.6f}")
    print(f"  Median: {valid_scores.median():.6f}")
    print(f"  Q1:     {valid_scores.quantile(0.25):.6f}")
    print(f"  Q3:     {valid_scores.quantile(0.75):.6f}")
    
    # Group by split_type
    if "split_type" not in df.columns:
        print("WARNING: 'split_type' column not found")
        return
    
    print("\n" + "="*70)
    print("STATISTICS BY SPLIT TYPE")
    print("="*70)
    
    summary_rows = []
    
    for split_type in sorted(df["split_type"].unique()):
        subset = df[df["split_type"] == split_type]["clip_score"].dropna()
        
        if len(subset) > 0:
            stats_dict = {
                "split_type": split_type,
                "count": len(subset),
                "mean": subset.mean(),
                "std": subset.std(),
                "min": subset.min(),
                "q25": subset.quantile(0.25),
                "median": subset.median(),
                "q75": subset.quantile(0.75),
                "max": subset.max(),
            }
            summary_rows.append(stats_dict)
            
            print(f"\n{split_type} ({len(subset):,} samples):")
            print(f"  Mean:   {subset.mean():.6f}")
            print(f"  Std:    {subset.std():.6f}")
            print(f"  Min:    {subset.min():.6f}")
            print(f"  Q1:     {subset.quantile(0.25):.6f}")
            print(f"  Median: {subset.median():.6f}")
            print(f"  Q3:     {subset.quantile(0.75):.6f}")
            print(f"  Max:    {subset.max():.6f}")
    
    # Also group by class_label if available
    if "class_label" in df.columns:
        print("\n" + "="*70)
        print("STATISTICS BY CLASS LABEL")
        print("="*70)
        
        for class_label in sorted(df["class_label"].unique()):
            subset = df[df["class_label"] == class_label]["clip_score"].dropna()
            if len(subset) > 0:
                print(f"\n{class_label} ({len(subset):,} samples):")
                print(f"  Mean:   {subset.mean():.6f}")
                print(f"  Std:    {subset.std():.6f}")
                print(f"  Min:    {subset.min():.6f}")
                print(f"  Max:    {subset.max():.6f}")
    
    # Save summary table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        os.makedirs(results_dir, exist_ok=True)
        summary_df.to_csv(output_path, index=False, float_format="%.6f")
        print(f"\n" + "="*70)
        print(f"\nSaved summary to {output_path}")
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize CLIP scores")
    parser.add_argument(
        "--input", default="clip_scores_results.csv",
        help="Input results CSV (in clip_scoring/)"
    )
    parser.add_argument(
        "--output", default="clip_summary_table.csv",
        help="Output summary CSV (in clip_scoring/)"
    )
    args = parser.parse_args()
    
    summarize_clip_scores(
        input_csv=args.input,
        output_csv=args.output,
    )
    print("Done!")
