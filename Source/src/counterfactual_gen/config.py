"""
Configuration file for hate speech dataset construction.
"""

# Dataset parameters
DATASET_NAME = "ucberkeley-dlab/measuring-hate-speech"
DATASET_CONFIG = "default"
TOTAL_SAMPLES = 6000  # 8 classes × 750 samples
SAMPLES_PER_CLASS = 750

# Class definitions
HATE_CLASSES = {
    'hate_race': {
        'description': 'Hate speech targeting race/ethnicity',
        'target_groups': ['race/ethnicity'],
        'polarity': 'hate'
    },
    'hate_religion': {
        'description': 'Hate speech targeting religion',
        'target_groups': ['religion'],
        'polarity': 'hate'
    },
    'hate_gender': {
        'description': 'Hate speech targeting gender/sexual orientation',
        'target_groups': ['gender', 'sexual_orientation'],
        'polarity': 'hate'
    },
    'hate_other': {
        'description': 'Hate speech targeting other groups (nationality, age, disability, politics)',
        'target_groups': ['national_origin/citizenship', 'age', 'disability', 'political_ideology'],
        'polarity': 'hate'
    }
}

NON_HATE_CLASSES = {
    'offensive_non_hate': {
        'description': 'Offensive language without hate (profanity, crude language)',
        'polarity': 'non-hate'
    },
    'neutral_discussion': {
        'description': 'Neutral discussion mentioning identity groups',
        'polarity': 'non-hate'
    },
    'counter_speech': {
        'description': 'Speech countering hate or supporting marginalized groups',
        'polarity': 'non-hate'
    },
    'ambiguous': {
        'description': 'Ambiguous cases with multiple targets or unclear intent',
        'polarity': 'non-hate'
    }
}

ALL_CLASSES = {**HATE_CLASSES, **NON_HATE_CLASSES}

# Target group mapping (from dataset to our taxonomy)
TARGET_GROUP_MAPPING = {
    'race': 'race/ethnicity',
    'ethnicity': 'race/ethnicity',
    'religion': 'religion',
    'gender': 'gender',
    'sex': 'gender',
    'sexual_orientation': 'sexual_orientation',
    'lgbtq': 'sexual_orientation',
    'nationality': 'national_origin/citizenship',
    'national_origin': 'national_origin/citizenship',
    'citizenship': 'national_origin/citizenship',
    'age': 'age',
    'disability': 'disability',
    'political': 'political_ideology',
    'politics': 'political_ideology',
}

# Filtering thresholds
MIN_TEXT_LENGTH = 10  # words
MAX_TEXT_LENGTH = 200  # words
HATE_SCORE_THRESHOLD_HIGH = 0.6  # Clear hate speech
HATE_SCORE_THRESHOLD_MED = 0.35  # Medium hate score (offensive but not clear hate)
HATE_SCORE_THRESHOLD_LOW = 0.2  # Low hate score (neutral to counter-speech)
MIN_CONFIDENCE = 3  # Minimum number of annotators agreeing (out of typically 5-10)

# Quality control  
DUPLICATE_SIMILARITY_THRESHOLD = 0.95  # Jaccard similarity for near-duplicate detection (relaxed)
MIN_HATE_SCORE_FOR_HATE_CLASS = 0.6
MAX_HATE_SCORE_FOR_NON_HATE_CLASS = 0.35

# Output paths
OUTPUT_DIR = "/home/vslinux/Documents/research/major-project/src/counterfactual_gen"
OUTPUT_CSV = f"{OUTPUT_DIR}/hate_speech_dataset_6k.csv"
OUTPUT_STATS = f"{OUTPUT_DIR}/dataset_statistics.json"
OUTPUT_README = f"{OUTPUT_DIR}/README.md"

# Random seed for reproducibility
RANDOM_SEED = 42
