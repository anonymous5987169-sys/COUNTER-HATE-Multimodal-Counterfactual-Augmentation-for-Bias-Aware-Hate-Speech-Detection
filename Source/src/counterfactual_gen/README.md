# Hate Speech Dataset (6,000 Samples - 8 Classes)

## Overview

This dataset contains 6,000 samples of social media text, perfectly balanced across 8 distinct classes related to hate speech detection. The dataset was constructed from the "Measuring Hate Speech" dataset by Kennedy et al. (2020) and systematically processed to create a clean, balanced resource for hate speech classification research.

## Dataset Statistics

- **Total Samples**: 6,000
- **Classes**: 8 (4 hate + 4 non-hate)
- **Samples per Class**: 750
- **Perfect Balance**: 3,000 hate | 3,000 non-hate
- **Source**: [Measuring Hate Speech (Kennedy et al., 2020)](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
- **Language**: English
- **Text Length**: 10-200 words (average: ~32 words)

## Class Taxonomy

### Hate Speech Classes (4 classes - 3,000 samples)

#### 1. hate_race
- **Description**: Hate speech targeting race/ethnicity
- **Target Groups**: Race/ethnicity
- **Sample Count**: 750

#### 2. hate_religion  
- **Description**: Hate speech targeting religion
- **Target Groups**: Religion
- **Sample Count**: 750

#### 3. hate_gender
- **Description**: Hate speech targeting gender/sexual orientation
- **Target Groups**: Gender, Sexual orientation
- **Sample Count**: 750

#### 4. hate_other
- **Description**: Hate speech targeting other identity groups
- **Target Groups**: National origin/citizenship, Age, Disability
- **Sample Count**: 750

### Non-Hate Classes (4 classes - 3,000 samples)

#### 5. offensive_non_hate
- **Description**: Offensive language without targeted hate
- **Characteristics**: Profanity, crude language, general offensive content
- **Note**: Contains offensive words but doesn't target specific identity groups with hate
- **Sample Count**: 750

#### 6. neutral_discussion
- **Description**: Neutral discussion mentioning identity groups
- **Characteristics**: Factual discourse, neutral commentary about social issues
- **Note**: Mentions identity groups without expressing hate or strong sentiment
- **Sample Count**: 750

#### 7. counter_speech
- **Description**: Speech countering hate or supporting marginalized groups
- **Characteristics**: Positive sentiment, supportive language, anti-hate messaging
- **Note**: Actively opposes hate speech or supports targeted communities
- **Sample Count**: 750

#### 8. ambiguous
- **Description**: Ambiguous cases with multiple targets or unclear intent
- **Characteristics**: Samples mentioning multiple identity groups, borderline cases, or no clear target
- **Note**: Captures complex cases that don't fit cleanly into other categories
- **Sample Count**: 750

## Dataset Construction Methodology

### Source Dataset
- **Name**: Measuring Hate Speech
- **Authors**: Kennedy et al. (2020)
- **Size**: 135,556 total samples
- **Features**: Multi-dimensional hate speech annotations including hate scores, sentiment, and target group labels

### Processing Pipeline

1. **Classification & Filtering**
   - Mapped source annotations to 8-class taxonomy
   - Filtered for samples with single primary target group (except ambiguous class)
   - Text length filtering (10-200 words) to avoid ambiguity
   - Quality thresholds:
     - Hate classes: hate_score ≥ 0.6
     - Offensive non-hate: 0.35 ≤ hate_score < 0.6
     - Non-hate classes: hate_score < 0.35
     - Ambiguous: Multiple targets or no targets with hate_score < 0.6

2. **Deduplication**
   - Removed exact text duplicates using MD5 hashing
   - Reduced from 98,322 to 30,597 unique samples
   - Removed 67,725 duplicate texts (68.9% deduplication rate)

3. **Class Balancing**
   - Target: 750 samples per class
   - Method: High-confidence sampling (sorted by confidence scores)
   - Minimal oversampling: Only 1 sample for hate_religion class (749→750)

4. **Metadata Assignment**
   - Unique sample IDs: Format `HS_{CLASS}_{INDEX}`
   - Class labels, target groups, polarity markers
   - Hate scores and confidence scores preserved

## File Structure

```
hate_speech_dataset_6k.csv          # Main dataset file (6,000 samples)
dataset_statistics.json              # Statistical summary
README.md                            # This file
config.py                            # Configuration parameters
hate_speech_dataset_builder.py      # Processing pipeline
utils.py                             # Utility functions
example_usage.py                     # Usage demonstration script
```

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique identifier (HS_{CLASS}_{INDEX}) |
| `text` | string | Social media text content |
| `class_label` | string | One of 8 class labels |
| `target_group` | string | Primary/multiple target identity group(s) |
| `polarity` | string | "hate" or "non-hate" |
| `hate_score` | float | Original hate speech score from source |
| `confidence` | float | Classification confidence score |

## Usage Guidelines

### Loading the Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('hate_speech_dataset_6k.csv')

# View class distribution
print(df['class_label'].value_counts())

# Result:
# Each class has exactly 750 samples

# Filter by polarity
hate_samples = df[df['polarity'] == 'hate']        # 3,000 samples
non_hate_samples = df[df['polarity'] == 'non-hate']  # 3,000 samples
```

### Train/Val/Test Split Recommendation

```python
from sklearn.model_selection import train_test_split

# Stratified split to maintain class balance
# Target ratio: 70/15/15
# Canonical realized ratio: 69.99/15.00/15.01 (train/val/test = 4158/891/892)
train, temp = train_test_split(df, test_size=0.30, stratify=df['class_label'], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp['class_label'], random_state=42)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
# Canonical split counts are persisted in data/splits/canonical_splits.json
```

## Ethical Considerations

### Content Warning
⚠️ **This dataset contains offensive, hateful, and disturbing content including slurs, threats, and profanity.** It is intended strictly for research purposes in hate speech detection and content moderation.

### Intended Use
- Training hate speech classification models
- Research on online toxicity and harassment
- Development of content moderation systems
- Academic studies on hate speech patterns
- Multi-class hate speech detection research

### Prohibited Uses
- Training models to generate hate speech
- Harassment or targeting of individuals/groups
- Amplification of hateful content
- Any use that could cause harm to protected groups

### Limitations
- **Temporal**: Data from social media posts may not reflect current language trends
- **Platform Bias**: Primarily Twitter/social media language patterns
- **Annotation Bias**: Inherited from source dataset's annotation methodology
- **Coverage**: Limited to English language content
- **Ambiguous Class**: May contain heterogeneous samples; requires careful consideration

## Class Balance & Distribution

```
Perfect 8-Class Balance (750 samples each):

├── Hate Speech (3,000 samples - 50.0%)
│   ├── hate_race: 750
│   ├── hate_religion: 750
│   ├── hate_gender: 750
│   └── hate_other: 750
└── Non-Hate (3,000 samples - 50.0%)
    ├── offensive_non_hate: 750
    ├── neutral_discussion: 750
    ├── counter_speech: 750
    └── ambiguous: 750
```

## Example Samples

### Hate Speech Examples

**hate_race**: "I wanna shoot one of them niggas that wear fake bulletproof vests 😭"

**hate_religion**: "Kick out all the pisslams staying illigally in the United States..."

**hate_gender**: "fr diva gays are the worst types of faggots. take some phenobarbital..."

**hate_other**: "Dig burn PITS and treat them THE INVADING NAZIS ALL OVER EUROPE..."

### Non-Hate Examples

**offensive_non_hate**: "Woman need to remove themselves as organ donors. This is utterly appalling"

**neutral_discussion**: "There's a church near where in live that used to be unoccupied and so it was transformed into a Sikh temple..."

**counter_speech**: "Trans students are not a threat. This move means school districts can continue to allow trans students to participate..."

**ambiguous**: "I love you beautiful black Queen. Bless all my beautiful black sistas around the world. YOU ARE WORTH IT!"

## Citation

If you use this dataset in your research, please cite both this dataset and the original source:

### This Dataset
```
@dataset{hate_speech_balanced_2026,
  title={Balanced Hate Speech Dataset: 8-Class Taxonomy with 6,000 Samples},
  author={[Your Name]},
  year={2026},
  note={Derived from Measuring Hate Speech (Kennedy et al., 2020)}
}
```

### Source Dataset
```
@article{kennedy2020measuring,
  title={The measuring hate speech corpus: Leveraging rasch measurement theory for data perspectivism},
  author={Kennedy, Brendan and Atari, Mohammad and Davani, Aida M and Hoover, Joe and Omrani, Ali and Graham, Jesse and Dehghani, Morteza},
  journal={Proceedings of the 1st Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP},
  year={2020}
}
```

## License

This dataset inherits the license from the source "Measuring Hate Speech" dataset. Please refer to the original dataset's license for terms of use.

## Contact & Questions

For questions, issues, or suggestions regarding this dataset, please open an issue in the repository or contact the maintainers.

---

**Version**: 2.0 (Complete 8-class version)
**Last Updated**: January 2026  
**Format**: CSV (UTF-8 encoded)  
**Total Samples**: 6,000 (perfectly balanced)
