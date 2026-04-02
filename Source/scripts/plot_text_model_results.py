#!/usr/bin/env python3
"""
Plot text model results from Table 2.
Compares TF-IDF and MiniLM models across nCF and CF conditions.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from Table 2
data = {
    "Model": [
        "TF-IDF\n+ LogReg",
        "TF-IDF\n+ SVM",
        "TF-IDF\n+ RF",
        "MiniLM\n+ LogReg",
        "MiniLM\n+ SVM",
        "MiniLM\n+ MLP",
    ],
    "nCF_F1": [0.793, 0.828, 0.825, 0.863, 0.855, 0.863],
    "CF_F1": [0.786, 0.810, 0.830, 0.873, 0.871, 0.946],
    "nCF_FPR": [0.234, 0.212, 0.288, 0.252, 0.273, 0.237],
    "CF_FPR": [0.291, 0.266, 0.367, 0.246, 0.252, 0.059],
    "Feature": ["TF-IDF", "TF-IDF", "TF-IDF", "MiniLM", "MiniLM", "MiniLM"],
}

df = pd.DataFrame(data)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Text Model Results: nCF vs CF on 900-Sample Test Set", fontsize=16, fontweight="bold", y=1.00)

# Subplot 1: F1 Score Comparison
ax1 = axes[0]
x = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x - width/2, df["nCF_F1"], width, label="nCF (original only)", color="#4CAF50", alpha=0.8)
bars2 = ax1.bar(x + width/2, df["CF_F1"], width, label="CF (with counterfactuals)", color="#FF6B6B", alpha=0.8)

ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
ax1.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
ax1.set_title("F1 Score Comparison", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(df["Model"], fontsize=10)
ax1.set_ylim([0.75, 0.97])
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height,
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

# Highlight best F1 (MiniLM + MLP with CF = 0.946)
best_idx = df["CF_F1"].idxmax()
ax1.annotate(
    "Best: 0.946",
    xy=(best_idx, df.loc[best_idx, "CF_F1"]),
    xytext=(best_idx+0.5, 0.92),
    fontsize=10, fontweight="bold", color="#FF6B6B",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
    arrowprops=dict(arrowstyle="->", color="#FF6B6B", lw=2),
)

# Subplot 2: FPR Comparison (Lower is better)
ax2 = axes[1]
bars3 = ax2.bar(x - width/2, df["nCF_FPR"], width, label="nCF (original only)", color="#2196F3", alpha=0.8)
bars4 = ax2.bar(x + width/2, df["CF_FPR"], width, label="CF (with counterfactuals)", color="#FF9800", alpha=0.8)

ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
ax2.set_ylabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
ax2.set_title("FPR Comparison (Lower is Better)", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(df["Model"], fontsize=10)
ax2.set_ylim([0.0, 0.40])
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., height,
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

# Highlight best FPR (MiniLM + MLP with CF = 0.059)
best_fpr_idx = df["CF_FPR"].idxmin()
ax2.annotate(
    "Best: 0.059",
    xy=(best_fpr_idx, df.loc[best_fpr_idx, "CF_FPR"]),
    xytext=(best_fpr_idx+0.5, 0.10),
    fontsize=10, fontweight="bold", color="#FF9800",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
    arrowprops=dict(arrowstyle="->", color="#FF9800", lw=2),
)

plt.tight_layout()
plt.savefig("/home/vslinux/Documents/research/major-project/outputs/plots/text_model_results.png", dpi=300, bbox_inches="tight")
print("✓ Saved: outputs/plots/text_model_results.png")
plt.close()

# ----- Additional Visualization: Effect of Counterfactuals -----

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Impact of Counterfactual Data on Model Performance", fontsize=16, fontweight="bold", y=1.00)

# Subplot 1: F1 Change
ax3 = axes2[0]
f1_change = (df["CF_F1"] - df["nCF_F1"]) * 100  # percentage points
colors_f1 = ["#4CAF50" if x > 0 else "#FF6B6B" for x in f1_change]

bars5 = ax3.barh(df["Model"], f1_change, color=colors_f1, alpha=0.8)
ax3.set_xlabel("F1 Change (percentage points)", fontsize=12, fontweight="bold")
ax3.set_title("F1 Score: CF vs nCF", fontsize=13, fontweight="bold")
ax3.axvline(x=0, color="black", linestyle="-", linewidth=1)
ax3.grid(axis="x", alpha=0.3, linestyle="--")

# Add value labels
for i, (bar, val) in enumerate(zip(bars5, f1_change)):
    x_pos = val + (0.3 if val > 0 else -0.3)
    ax3.text(x_pos, i, f"{val:+.1f}pp", ha="left" if val > 0 else "right", va="center", fontweight="bold", fontsize=9)

# Subplot 2: FPR Change (negative is better - lower FPR is better)
ax4 = axes2[1]
fpr_change = (df["CF_FPR"] - df["nCF_FPR"]) * 100  # percentage points
colors_fpr = ["#4CAF50" if x < 0 else "#FF6B6B" for x in fpr_change]  # Green if FPR decreased

bars6 = ax4.barh(df["Model"], fpr_change, color=colors_fpr, alpha=0.8)
ax4.set_xlabel("FPR Change (percentage points)", fontsize=12, fontweight="bold")
ax4.set_title("FPR: CF vs nCF (↓ is better)", fontsize=13, fontweight="bold")
ax4.axvline(x=0, color="black", linestyle="-", linewidth=1)
ax4.grid(axis="x", alpha=0.3, linestyle="--")

# Add value labels
for i, (bar, val) in enumerate(zip(bars6, fpr_change)):
    x_pos = val + (0.5 if val > 0 else -0.5)
    ax4.text(x_pos, i, f"{val:+.1f}pp", ha="left" if val > 0 else "right", va="center", fontweight="bold", fontsize=9)

plt.tight_layout()
plt.savefig("/home/vslinux/Documents/research/major-project/outputs/plots/text_model_cf_impact.png", dpi=300, bbox_inches="tight")
print("✓ Saved: outputs/plots/text_model_cf_impact.png")
plt.close()

# ----- Summary Statistics -----

print("\n" + "="*70)
print("TEXT MODEL RESULTS SUMMARY")
print("="*70)

print("\n📊 F1 Score Statistics:")
print(f"  TF-IDF best:  {df[df['Feature']=='TF-IDF']['CF_F1'].max():.3f} ({df[df['Feature']=='TF-IDF']['CF_F1'].idxmax()} — RF)")
print(f"  MiniLM best:  {df[df['Feature']=='MiniLM']['CF_F1'].max():.3f} ({df[df['Feature']=='MiniLM']['CF_F1'].idxmax()} — MLP)  ⭐")
print(f"  Overall best: {df['CF_F1'].max():.3f}")

print("\n📊 FPR Statistics (lower is better):")
print(f"  TF-IDF best:  {df[df['Feature']=='TF-IDF']['CF_FPR'].min():.3f} (LogReg)")
print(f"  MiniLM best:  {df[df['Feature']=='MiniLM']['CF_FPR'].min():.3f} (MLP)  ⭐")
print(f"  Overall best: {df['CF_FPR'].min():.3f}")

print("\n📈 Counterfactual Impact (CF - nCF):")
print(f"  F1 improvement: {f1_change.mean():+.1f}pp avg ({f1_change.min():.1f}pp to {f1_change.max():.1f}pp)")
print(f"  FPR change:    {fpr_change.mean():+.1f}pp avg ({fpr_change.min():.1f}pp to {fpr_change.max():.1f}pp)")
print(f"    → {sum(fpr_change < 0)}/6 models improved FPR (lower is better)")

print("\n" + "="*70)
