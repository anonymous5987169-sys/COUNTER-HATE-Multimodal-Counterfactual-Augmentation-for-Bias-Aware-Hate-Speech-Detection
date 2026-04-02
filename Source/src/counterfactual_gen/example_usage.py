"""
Example Usage Script for Hate Speech Dataset

This script demonstrates how to load and use the hate speech dataset
for machine learning tasks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('hate_speech_dataset_6k.csv')

print(f"\n{'='*80}")
print("DATASET OVERVIEW")
print(f"{'='*80}")
print(f"Total samples: {len(df)}")
print(f"Number of classes: {df['class_label'].nunique()}")
print(f"Columns: {', '.join(df.columns)}")

# Class distribution
print(f"\n{'='*80}")
print("CLASS DISTRIBUTION")
print(f"{'='*80}")
class_dist = df['class_label'].value_counts().sort_index()
for class_name, count in class_dist.items():
    percentage = (count / len(df)) * 100
    print(f"{class_name:20s}: {count:4d} samples ({percentage:5.2f}%)")

# Polarity distribution
print(f"\n{'='*80}")
print("POLARITY DISTRIBUTION (Binary Classification)")
print(f"{'='*80}")
polarity_dist = df['polarity'].value_counts()
for polarity, count in polarity_dist.items():
    percentage = (count / len(df)) * 100
    print(f"{polarity:10s}: {count:4d} samples ({percentage:5.2f}%)")

# Target group distribution
print(f"\n{'='*80}")
print("TARGET GROUP DISTRIBUTION")
print(f"{'='*80}")
target_dist = df['target_group'].value_counts().sort_values(ascending=False)
for target, count in target_dist.items():
    percentage = (count / len(df)) * 100
    print(f"{target:30s}: {count:4d} samples ({percentage:5.2f}%)")

# Text statistics
print(f"\n{'='*80}")
print("TEXT STATISTICS")
print(f"{'='*80}")
df['word_count'] = df['text'].str.split().str.len()
print(f"Average length: {df['word_count'].mean():.1f} words")
print(f"Median length:  {df['word_count'].median():.1f} words")
print(f"Min length:     {df['word_count'].min()} words")
print(f"Max length:     {df['word_count'].max()} words")
print(f"Std deviation:  {df['word_count'].std():.1f} words")

# Train/Val/Test split example
print(f"\n{'='*80}")
print("TRAIN/VAL/TEST SPLIT (70/15/15)")
print(f"{'='*80}")

# Stratified split
train, temp = train_test_split(
    df, 
    test_size=0.30, 
    stratify=df['class_label'], 
    random_state=42
)

val, test = train_test_split(
    temp, 
    test_size=0.50, 
    stratify=temp['class_label'], 
    random_state=42
)

print(f"Training set:   {len(train):4d} samples ({len(train)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val):4d} samples ({len(val)/len(df)*100:.1f}%)")
print(f"Test set:       {len(test):4d} samples ({len(test)/len(df)*100:.1f}%)")

# Verify stratification
print(f"\nClass balance verification:")
print(f"{'Class':<20s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
print("-" * 50)
for class_name in sorted(df['class_label'].unique()):
    train_count = (train['class_label'] == class_name).sum()
    val_count = (val['class_label'] == class_name).sum()
    test_count = (test['class_label'] == class_name).sum()
    print(f"{class_name:<20s} {train_count:>8d} {val_count:>8d} {test_count:>8d}")

# Show example samples
print(f"\n{'='*80}")
print("EXAMPLE SAMPLES (one from each class)")
print(f"{'='*80}")

for class_name in sorted(df['class_label'].unique()):
    sample = df[df['class_label'] == class_name].iloc[0]
    print(f"\n{class_name.upper()}:")
    print(f"  ID: {sample['sample_id']}")
    print(f"  Text: {sample['text'][:150]}...")
    print(f"  Target: {sample['target_group']}")
    print(f"  Polarity: {sample['polarity']}")
    print(f"  Hate Score: {sample['hate_score']:.2f}")

print(f"\n{'='*80}")
print("READY FOR MODEL TRAINING!")
print(f"{'='*80}")
print("\nSuggested next steps:")
print("1. Implement text preprocessing (tokenization, cleaning)")
print("2. Create feature vectors (TF-IDF, Word2Vec, or BERT embeddings)")
print("3. Train baseline classifier (Logistic Regression, SVM)")
print("4. Fine-tune transformer model (BERT, RoBERTa)")
print("5. Evaluate on test set and analyze errors")
print("\nGood luck with your hate speech detection project!")
