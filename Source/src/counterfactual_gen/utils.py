"""
Utility functions for hate speech dataset construction.
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Set, Tuple
import hashlib


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags (keep the text part)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Text string
        
    Returns:
        Word count
    """
    return len(text.split())


def is_valid_length(text: str, min_words: int, max_words: int) -> bool:
    """
    Check if text length is within valid range.
    
    Args:
        text: Text string
        min_words: Minimum word count
        max_words: Maximum word count
        
    Returns:
        True if valid length, False otherwise
    """
    word_count = count_words(text)
    return min_words <= word_count <= max_words


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Jaccard similarity score (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def find_duplicates(texts: List[str], threshold: float = 0.85) -> Set[int]:
    """
    Find exact duplicate texts.
    
    Args:
        texts: List of text strings
        threshold: Not used, kept for compatibility
        
    Returns:
        Set of indices to remove (duplicates)
    """
    to_remove = set()
    n = len(texts)
    
    # Create text hashes for exact duplicate detection
    text_hashes = {}
    for i, text in enumerate(texts):
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        if text_hash in text_hashes:
            to_remove.add(i)
        else:
            text_hashes[text_hash] = i
    
    return to_remove


def calculate_class_distribution(labels: List[str]) -> Dict[str, int]:
    """
    Calculate distribution of class labels.
    
    Args:
        labels: List of class labels
        
    Returns:
        Dictionary mapping class to count
    """
    return dict(Counter(labels))


def generate_sample_id(class_label: str, index: int) -> str:
    """
    Generate unique sample ID.
    
    Args:
        class_label: Class label
        index: Sample index within class
        
    Returns:
        Unique sample ID
    """
    # Convert class label to uppercase abbreviation
    class_abbrev = class_label.upper().replace('_', '')
    return f"HS_{class_abbrev}_{index:04d}"


def calculate_statistics(data: Dict) -> Dict:
    """
    Calculate dataset statistics.
    
    Args:
        data: Dictionary containing dataset information
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_samples': len(data.get('text', [])),
        'class_distribution': {},
        'target_group_distribution': {},
        'polarity_distribution': {},
        'text_length_stats': {},
        'hate_score_stats': {}
    }
    
    # Class distribution
    if 'class_label' in data:
        stats['class_distribution'] = calculate_class_distribution(data['class_label'])
    
    # Target group distribution
    if 'target_group' in data:
        stats['target_group_distribution'] = calculate_class_distribution(data['target_group'])
    
    # Polarity distribution
    if 'polarity' in data:
        stats['polarity_distribution'] = calculate_class_distribution(data['polarity'])
    
    # Text length statistics
    if 'text' in data:
        lengths = [count_words(text) for text in data['text']]
        stats['text_length_stats'] = {
            'mean': float(np.mean(lengths)),
            'median': float(np.median(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'std': float(np.std(lengths))
        }
    
    # Hate score statistics by class
    if 'hate_score' in data and 'class_label' in data:
        hate_scores_by_class = {}
        for class_label in set(data['class_label']):
            class_scores = [score for score, label in zip(data['hate_score'], data['class_label']) 
                          if label == class_label]
            if class_scores:
                hate_scores_by_class[class_label] = {
                    'mean': float(np.mean(class_scores)),
                    'median': float(np.median(class_scores)),
                    'min': float(np.min(class_scores)),
                    'max': float(np.max(class_scores))
                }
        stats['hate_score_stats'] = hate_scores_by_class
    
    return stats


def validate_dataset(df, expected_samples: int, expected_per_class: int) -> Tuple[bool, List[str]]:
    """
    Validate the final dataset.
    
    Args:
        df: DataFrame containing the dataset
        expected_samples: Expected total number of samples
        expected_per_class: Expected samples per class
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check total samples
    if len(df) != expected_samples:
        errors.append(f"Expected {expected_samples} samples, got {len(df)}")
    
    # Check samples per class
    class_counts = df['class_label'].value_counts()
    for class_label, count in class_counts.items():
        if count != expected_per_class:
            errors.append(f"Class '{class_label}' has {count} samples, expected {expected_per_class}")
    
    # Check for duplicate sample IDs
    if df['sample_id'].duplicated().any():
        errors.append("Duplicate sample IDs found")
    
    # Check for missing values
    required_columns = ['sample_id', 'text', 'class_label', 'target_group', 'polarity']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
        elif df[col].isna().any():
            errors.append(f"Missing values found in column: {col}")
    
    # Check polarity consistency
    if 'polarity' in df.columns and 'class_label' in df.columns:
        for _, row in df.iterrows():
            expected_polarity = 'hate' if row['class_label'].startswith('hate_') else 'non-hate'
            if row['polarity'] != expected_polarity:
                errors.append(f"Polarity mismatch for sample {row.get('sample_id', 'unknown')}")
                break  # Just report first occurrence
    
    is_valid = len(errors) == 0
    return is_valid, errors
