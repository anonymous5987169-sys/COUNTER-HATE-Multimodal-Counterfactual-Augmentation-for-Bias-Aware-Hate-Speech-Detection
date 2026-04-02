"""
Hate Speech Dataset Builder

This script builds a balanced 8-class hate speech dataset from the
"Measuring Hate Speech" dataset by Kennedy et al. (2020).
"""

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import config
import utils


def load_source_dataset():
    """Load the Measuring Hate Speech dataset from Hugging Face."""
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    
    # Convert to pandas DataFrame
    df = dataset['train'].to_pandas()
    print(f"Loaded {len(df)} samples from source dataset")
    
    return df


def map_target_group(row):
    """
    Map target groups from the dataset to our taxonomy.
    
    The dataset has these target columns:
    - target_race, target_religion, target_origin, target_gender,
    - target_sexuality, target_age, target_disability
    """
    target_groups = []
    
    # Check which target groups are mentioned (value > 0 or True)
    if row.get('target_race', 0) > 0:
        target_groups.append('race/ethnicity')
    if row.get('target_religion', 0) > 0:
        target_groups.append('religion')
    if row.get('target_origin', 0) > 0:
        target_groups.append('national_origin/citizenship')
    if row.get('target_gender', 0) > 0:
        target_groups.append('gender')
    if row.get('target_sexuality', 0) > 0:
        target_groups.append('sexual_orientation')
    if row.get('target_age', 0) > 0:
        target_groups.append('age')
    if row.get('target_disability', 0) > 0:
        target_groups.append('disability')
    
    return target_groups


def classify_sample(row):
    """
    Classify a sample into one of the 8 classes.
    
    Args:
        row: DataFrame row with annotations
        
    Returns:
        Tuple of (class_label, target_group, confidence) or None if sample doesn't fit
    """
    # Get hate speech score (0-1 scale)
    hate_score = row.get('hate_speech_score', 0)
    
    # Get sentiment
    sentiment = row.get('sentiment', 0)  # Negative, neutral, positive
    
    # Get target groups
    target_groups = map_target_group(row)
    
    # For 'ambiguous' class: Multiple target groups OR no target groups with low hate
    # This captures cases that don't fit cleanly into other categories
    if len(target_groups) == 0 and hate_score < 1.0:
        return ('ambiguous', 'multiple/none', 1 - abs(hate_score))
    
    if len(target_groups) > 1 and hate_score < config.MIN_HATE_SCORE_FOR_HATE_CLASS:
        # Multiple targets but not hateful enough - mark as ambiguous
        return ('ambiguous', 'multiple/none', 1 - hate_score)
    
    # Filter: Must have single primary target group for other classes
    if len(target_groups) != 1:
        return None
    
    target_group = target_groups[0]
    
    # Classify into hate classes (high hate score)
    if hate_score >= config.MIN_HATE_SCORE_FOR_HATE_CLASS:
        # Determine which hate class based on target group
        if target_group == 'race/ethnicity':
            return ('hate_race', target_group, hate_score)
        elif target_group == 'religion':
            return ('hate_religion', target_group, hate_score)
        elif target_group in ['gender', 'sexual_orientation']:
            return ('hate_gender', target_group, hate_score)
        elif target_group in ['national_origin/citizenship', 'age', 'disability']:
            return ('hate_other', target_group, hate_score)
    
    # Offensive but not hate (medium hate score range)
    elif config.HATE_SCORE_THRESHOLD_MED <= hate_score < config.MIN_HATE_SCORE_FOR_HATE_CLASS:
        return ('offensive_non_hate', target_group, hate_score)
    
    # Low/no hate speech (classify by sentiment)
    elif hate_score < config.HATE_SCORE_THRESHOLD_MED:
        # Counter-speech (positive sentiment + mentions target group)
        if sentiment > 0.2:
            return ('counter_speech', target_group, 1 - hate_score)
        
        # Neutral discussion about target group
        else:
            return ('neutral_discussion', target_group, 1 - hate_score)
    
    return None


def process_and_filter_data(df):
    """
    Process and filter the source dataset.
    
    Args:
        df: Source DataFrame
        
    Returns:
        Processed DataFrame with classifications
    """
    print("\nProcessing and filtering data...")
    
    processed_data = {
        'text': [],
        'class_label': [],
        'target_group': [],
        'hate_score': [],
        'confidence': [],
        'word_count': []
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        # Get text
        text = row.get('text', '')
        if not isinstance(text, str) or not text.strip():
            continue
        
        # Clean text
        text = utils.clean_text(text)
        
        # Check text length
        if not utils.is_valid_length(text, config.MIN_TEXT_LENGTH, config.MAX_TEXT_LENGTH):
            continue
        
        # Classify sample
        result = classify_sample(row)
        if result is None:
            continue
        
        class_label, target_group, confidence = result
        hate_score = row.get('hate_speech_score', 0)
        
        processed_data['text'].append(text)
        processed_data['class_label'].append(class_label)
        processed_data['target_group'].append(target_group)
        processed_data['hate_score'].append(hate_score)
        processed_data['confidence'].append(confidence)
        processed_data['word_count'].append(utils.count_words(text))
    
    processed_df = pd.DataFrame(processed_data)
    print(f"Processed {len(processed_df)} valid samples")
    
    return processed_df


def add_unrelated_samples(processed_df, source_df, needed_samples):
    """
    Add 'unrelated' class samples (normal speech unrelated to identity groups).
    
    Args:
        processed_df: Already processed DataFrame
        source_df: Source dataset
        needed_samples: Number of unrelated samples needed
        
    Returns:
        Updated DataFrame with unrelated samples
    """
    print(f"\nAdding {needed_samples} 'unrelated' class samples...")
    
    unrelated_data = []
    
    for idx, row in source_df.iterrows():
        if len(unrelated_data) >= needed_samples:
            break
        
        # Get target groups
        target_groups = map_target_group(row)
        
        # Must have NO target groups and low hate score
        if len(target_groups) == 0 and row.get('hate_speech_score', 0) < 0.2:
            text = row.get('text', '')
            if not isinstance(text, str) or not text.strip():
                continue
            
            text = utils.clean_text(text)
            
            if utils.is_valid_length(text, config.MIN_TEXT_LENGTH, config.MAX_TEXT_LENGTH):
                unrelated_data.append({
                    'text': text,
                    'class_label': 'unrelated',
                    'target_group': 'none',
                    'hate_score': row.get('hate_speech_score', 0),
                    'confidence': 1 - row.get('hate_speech_score', 0),
                    'word_count': utils.count_words(text)
                })
    
    if unrelated_data:
        unrelated_df = pd.DataFrame(unrelated_data)
        processed_df = pd.concat([processed_df, unrelated_df], ignore_index=True)
        print(f"Added {len(unrelated_data)} unrelated samples")
    
    return processed_df


def remove_duplicates(df):
    """Remove duplicate and near-duplicate samples."""
    print("\nRemoving duplicates...")
    
    texts = df['text'].tolist()
    duplicates = utils.find_duplicates(texts, config.DUPLICATE_SIMILARITY_THRESHOLD)
    
    if duplicates:
        df = df.drop(index=list(duplicates)).reset_index(drop=True)
        print(f"Removed {len(duplicates)} duplicate samples")
    else:
        print("No duplicates found")
    
    return df


def balance_dataset(df):
    """
    Balance the dataset to have equal samples per class.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Balanced DataFrame with SAMPLES_PER_CLASS samples for each class
    """
    print("\nBalancing dataset...")
    
    class_counts = df['class_label'].value_counts()
    print("\nClass distribution before balancing:")
    for class_label, count in class_counts.items():
        print(f"  {class_label}: {count}")
    
    balanced_data = []
    
    for class_label in config.ALL_CLASSES.keys():
        class_df = df[df['class_label'] == class_label]
        
        if len(class_df) >= config.SAMPLES_PER_CLASS:
            # Sample with priority to higher confidence scores
            class_df = class_df.sort_values('confidence', ascending=False)
            sampled = class_df.head(config.SAMPLES_PER_CLASS)
        elif len(class_df) > 100:  # If we have at least 100 samples, oversample
            print(f"  INFO: Oversampling class '{class_label}' from {len(class_df)} to {config.SAMPLES_PER_CLASS} samples")
            sampled = class_df.sample(n=config.SAMPLES_PER_CLASS, replace=True, random_state=config.RANDOM_SEED)
        elif len(class_df) > 0:
            print(f"  WARNING: Insufficient samples for class '{class_label}' ({len(class_df)}), taking all available")
            sampled = class_df
        else:
            print(f"  WARNING: No samples for class '{class_label}'")
            sampled = pd.DataFrame()
        
        balanced_data.append(sampled)
    
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    print(f"\nBalanced dataset size: {len(balanced_df)}")
    print("\nClass distribution after balancing:")
    for class_label, count in balanced_df['class_label'].value_counts().items():
        print(f"  {class_label}: {count}")
    
    return balanced_df


def add_metadata(df):
    """Add metadata columns to the dataset."""
    print("\nAdding metadata...")
    
    # Add sample IDs
    sample_ids = []
    class_indices = {}
    
    for _, row in df.iterrows():
        class_label = row['class_label']
        if class_label not in class_indices:
            class_indices[class_label] = 0
        
        sample_id = utils.generate_sample_id(class_label, class_indices[class_label])
        sample_ids.append(sample_id)
        class_indices[class_label] += 1
    
    df['sample_id'] = sample_ids
    
    # Add polarity
    df['polarity'] = df['class_label'].apply(
        lambda x: config.ALL_CLASSES[x]['polarity']
    )
    
    # Reorder columns
    column_order = ['sample_id', 'text', 'class_label', 'target_group', 
                   'polarity', 'hate_score', 'confidence']
    df = df[column_order]
    
    return df


def save_dataset(df, stats):
    """Save the dataset and statistics."""
    print(f"\nSaving dataset to {config.OUTPUT_CSV}...")
    
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save CSV
    df.to_csv(config.OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Dataset saved: {config.OUTPUT_CSV}")
    
    # Save statistics
    with open(config.OUTPUT_STATS, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved: {config.OUTPUT_STATS}")


def main():
    """Main execution function."""
    print("="*80)
    print("Hate Speech Dataset Builder")
    print("="*80)
    
    # Set random seed
    np.random.seed(config.RANDOM_SEED)
    
    # Step 1: Load source dataset
    source_df = load_source_dataset()
    
    # Step 2: Process and filter data
    processed_df = process_and_filter_data(source_df)
    
    # Step 3: Remove duplicates
    processed_df = remove_duplicates(processed_df)
    
    # Step 4: Balance dataset (with oversampling for underrepresented classes)
    balanced_df = balance_dataset(processed_df)
    
    # Step 5: Add metadata
    final_df = add_metadata(balanced_df)
    
    # Step 6: Validate dataset
    print("\nValidating dataset...")
    is_valid, errors = utils.validate_dataset(
        final_df, 
        config.TOTAL_SAMPLES, 
        config.SAMPLES_PER_CLASS
    )
    
    if is_valid:
        print("✓ Dataset validation passed!")
    else:
        print("✗ Dataset validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    # Step 7: Calculate statistics
    print("\nCalculating statistics...")
    stats_dict = {
        'text': final_df['text'].tolist(),
        'class_label': final_df['class_label'].tolist(),
        'target_group': final_df['target_group'].tolist(),
        'polarity': final_df['polarity'].tolist(),
        'hate_score': final_df['hate_score'].tolist()
    }
    stats = utils.calculate_statistics(stats_dict)
    
    # Step 8: Save dataset and statistics
    save_dataset(final_df, stats)
    
    print("\n" + "="*80)
    print("Dataset construction complete!")
    print("="*80)
    print(f"\nFinal dataset: {len(final_df)} samples")
    print(f"Classes: {len(config.ALL_CLASSES)}")
    print(f"Samples per class: {config.SAMPLES_PER_CLASS}")
    print(f"\nOutput files:")
    print(f"  - Dataset: {config.OUTPUT_CSV}")
    print(f"  - Statistics: {config.OUTPUT_STATS}")
    
    return final_df, stats


if __name__ == "__main__":
    df, stats = main()
