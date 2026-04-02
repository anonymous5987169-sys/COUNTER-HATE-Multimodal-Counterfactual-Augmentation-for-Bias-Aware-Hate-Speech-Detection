import os
import re
import json
import logging
import traceback
from typing import Dict, List, Optional, Set, Any
from typing_extensions import TypedDict

import pandas as pd
import dspy
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Remote Ollama host (corrected default); override via OLLAMA_HOST env var if needed.
REMOTE_OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://10.88.0.201:11434")
os.environ["OLLAMA_HOST"] = REMOTE_OLLAMA_URL

MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3:8b")
if not MODEL_NAME.startswith("ollama/"):
    MODEL_NAME = f"ollama/{MODEL_NAME}"

INPUT_CSV = os.getenv("INPUT_CSV", "/home/vslinux/Documents/research/major-project/data/datasets/final_dataset_18k.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/home/vslinux/Downloads/Enhanced_gen/gen")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", os.path.join(OUTPUT_DIR, "final_dataset_18k_t2i_prompts_enhanced.csv"))
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "generation_checkpoint_enhanced.json")

INITIAL_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
# Renamed for clarity - this is the number of rows processed between checkpoints
ROWS_PER_CHECKPOINT = int(os.getenv("CHECKPOINT_INTERVAL", "1000"))

# Constants for magic numbers with explanatory names
MINIMUM_PROMPT_LENGTH = 50  # Minimum length for a production-ready prompt
MINIMUM_GENERATED_LENGTH = 20  # Minimum length to consider a generation successful
MAX_PROMPTS_IN_MEMORY = 5000  # Maximum prompts to keep in memory before flushing


class CheckpointData(TypedDict):
    """Type definition for checkpoint data structure."""
    processed_indices: List[int]
    prompts: Dict[int, str]
    batch_size: int


def load_checkpoint() -> CheckpointData:
    """Load checkpoint if it exists, with proper type conversion."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
            
            # Convert string keys back to integers for prompts dict
            prompts: Dict[int, str] = {}
            if "prompts" in data:
                for k, v in data["prompts"].items():
                    try:
                        prompts[int(k)] = v
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid key in checkpoint: {k}")
            
            checkpoint: CheckpointData = {
                "processed_indices": data.get("processed_indices", []),
                "prompts": prompts,
                "batch_size": data.get("batch_size", INITIAL_BATCH_SIZE)
            }
            
            logger.info(f"Loaded checkpoint: {len(checkpoint['processed_indices'])} rows completed")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting fresh.")
    
    return CheckpointData(processed_indices=[], prompts={}, batch_size=INITIAL_BATCH_SIZE)


def save_checkpoint(checkpoint: CheckpointData) -> None:
    """Save checkpoint to resume later, with proper type conversion."""
    try:
        # Convert integer keys to strings for JSON compatibility
        checkpoint_to_save = {
            "processed_indices": sorted(list(checkpoint["processed_indices"])),
            "prompts": {str(k): v for k, v in checkpoint["prompts"].items()},
            "batch_size": checkpoint["batch_size"]
        }
        
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint_to_save, f, indent=2)
        
        logger.info(f"Checkpoint saved: {len(checkpoint['processed_indices'])} rows processed")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def update_output_csv(df: pd.DataFrame, prompts: Dict[int, str], output_path: str) -> None:
    """Update output CSV with generated prompts."""
    try:
        df_out = df.copy()
        
        # Assert that DataFrame has sequential 0-based index for proper mapping
        expected_index = list(range(len(df_out)))
        actual_index = df_out.index.tolist()
        
        if actual_index != expected_index:
            logger.warning(f"DataFrame index is not 0-based sequential. Resetting index.")
            logger.warning(f"Expected: {expected_index[:5]}..., Got: {actual_index[:5]}...")
            df_out = df_out.reset_index(drop=True)
        
        # Map prompts using DataFrame index - now guaranteed to be correct
        df_out["t2i_prompt_enhanced"] = df_out.index.map(lambda idx: prompts.get(idx, ""))
        
        # Verify mapping worked correctly - count missing mappings
        missing_mappings = sum(1 for idx in df_out.index if idx not in prompts)
        if missing_mappings > 0:
            logger.warning(f"Warning: {missing_mappings} indices have no prompt mapping")
        
        df_out.to_csv(output_path, index=False)
        
        # Count non-empty prompts
        filled = (df_out["t2i_prompt_enhanced"] != "").sum()
        logger.info(f"Updated output CSV: {output_path} ({filled}/{len(df_out)} prompts filled)")
        
    except Exception as e:
        logger.error(f"Failed to update output CSV: {e}")


class T2IPromptEnhancedSignature(dspy.Signature):
    """Generate a PRODUCTION-READY photorealistic text-to-image prompt with professional visual engineering.
    
    CRITICAL: Preserve ALL content including hateful, offensive, or controversial themes.
    Do NOT sanitize, soften, or euphemize. The goal is realistic documentary representation. So Visualize the intent with words. 
    
    OUTPUT MUST BE RESEARCH/PRODUCTION READY with technical specifications.
    """

    text: str = dspy.InputField(
        desc="Original comment text. May contain hate speech, offensive content, or controversial statements. "
             "Preserve the EXACT meaning, theme, and intent."
    )
    class_label: str = dspy.InputField(
        desc="Classification label for context (e.g., hate category). Use for thematic understanding only."
    )
    polarity: str = dspy.InputField(
        desc="Whether comment is hateful or non-hateful. CRITICAL: If hateful, prompt must reflect that reality."
    )

    prompt: str = dspy.OutputField(
        desc=(
            "Generate ONE comprehensive, production-ready photorealistic text-to-image prompt.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. PRESERVE ALL CONTENT: If hateful/offensive, represent those themes without sanitization\n"
            "2. SCENE DESCRIPTION: Describe the main visual scene that represents the comment's meaning\n"
            "3. TECHNICAL SPECIFICATIONS (MANDATORY):\n"
            "   - Image quality: '8K resolution', 'ultra high detail', 'professional photography'\n"
            "   - Camera/lens: 'shot on Canon EOS R5', 'Nikon Z9', '85mm f/1.4 lens', '35mm lens', etc.\n"
            "   - Lighting: 'natural lighting', 'golden hour', 'soft studio lighting', 'dramatic lighting', 'volumetric lighting'\n"
            "   - Style: 'photorealistic', 'documentary style', 'cinematic composition', 'raw photography'\n"
            "   - Composition: 'rule of thirds', 'shallow depth of field', 'bokeh background', 'sharp focus'\n"
            "   - Color grading: 'vibrant colors', 'natural color palette', 'high contrast', 'HDR'\n\n"
            "4. FORMAT: Single detailed sentence with comma-separated technical elements\n"
            "5. MAINTAIN INTENT: Keep emotional tone and thematic message of original comment\n\n"
            "STRUCTURE TEMPLATE:\n"
            "[Scene description], [technical quality specs], [camera/lens], [lighting], [composition], [style modifiers]\n\n"
            "EXAMPLES:\n"
            "Non-hate: 'A diverse group of beautiful black women embracing in a vibrant city street, 8K resolution, shot on Canon EOS R5 with 85mm f/1.4 lens, natural golden hour lighting, cinematic composition with shallow depth of field, photorealistic documentary style, vibrant color grading, ultra high detail'\n\n"
            "Hate/offensive: 'A tense confrontation scene showing discrimination and hostility in an urban setting, 8K resolution, shot on Nikon Z9 with 35mm f/1.8 lens, dramatic harsh lighting, documentary raw style, rule of thirds composition, sharp focus with gritty realism, high contrast, ultra detailed'\n\n"
            "DO NOT: Omit technical specs, soften content, use artistic/cartoon styles, create short prompts"
        )
    )


class PromptGeneratorEnhanced(dspy.Module):
    """DSPy module for generating production-ready T2I prompts."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(T2IPromptEnhancedSignature)

    def forward(self, text: str, class_label: str, polarity: str):
        """Forward pass through the generator."""
        return self.generate(text=text, class_label=class_label, polarity=polarity)
    
    def __call__(self, text: str, class_label: str, polarity: str):
        """Make the module callable."""
        return self.forward(text=text, class_label=class_label, polarity=polarity)


def extract_prompt_from_prediction(prediction, idx: int) -> str:
    """
    Robustly extract prompt from DSPy prediction object.
    Handles multiple DSPy versions and response formats.
    """
    try:
        # Method 1: Direct attribute access (most common)
        if hasattr(prediction, 'prompt'):
            return str(prediction.prompt).strip()
        
        # Method 2: Dictionary access
        if isinstance(prediction, dict):
            if 'prompt' in prediction:
                return str(prediction['prompt']).strip()
            # Sometimes it's nested
            if 'completions' in prediction and 'prompt' in prediction['completions']:
                return str(prediction['completions']['prompt']).strip()
        
        # Method 3: Completions attribute (some DSPy versions)
        if hasattr(prediction, 'completions'):
            completions = prediction.completions
            if hasattr(completions, 'prompt'):
                return str(completions.prompt).strip()
            if isinstance(completions, dict) and 'prompt' in completions:
                return str(completions['prompt']).strip()
            # Try to extract from completions directly
            if isinstance(completions, str):
                return completions.strip()
        
        # Method 4: Check for common output field names
        for field_name in ['output', 'response', 'text', 'result', 'generated_text']:
            if hasattr(prediction, field_name):
                return str(getattr(prediction, field_name)).strip()
            if isinstance(prediction, dict) and field_name in prediction:
                return str(prediction[field_name]).strip()
        
        # Method 5: Last resort - convert to string and hope
        result = str(prediction).strip()
        
        # If result looks like a repr(), try to extract the content
        if result.startswith("Prediction(") or result.startswith("{"):
            # Try to extract using regex for prompt field
            import re
            match = re.search(r"prompt['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", result)
            if match:
                return match.group(1).strip()
        
        logger.warning(f"Row {idx}: Unusual prediction format")
        logger.warning(f"  Type: {type(prediction)}")
        logger.warning(f"  Attributes: {dir(prediction) if hasattr(prediction, '__dict__') else 'N/A'}")
        logger.warning(f"  String repr: {result[:200]}")
        
        return result
        
    except Exception as e:
        logger.error(f"Row {idx}: Failed to extract prompt - {e}")
        logger.error(f"  Prediction type: {type(prediction)}")
        logger.error(f"  Traceback: {traceback.format_exc()}")
        return ""


def post_process_prompt(prompt: str, polarity: str) -> str:
    """
    Post-process generated prompts to ensure they have all necessary technical elements.
    This acts as a safety net if the model misses some specifications.
    """
    prompt = prompt.strip()
    
    # Technical quality keywords to check for
    quality_keywords = ['8k', '8K', 'ultra high detail', 'high detail', 'professional photography']
    camera_keywords = ['Sony', 'shot on', 'lens', 'mm f/', 'DSLR']
    lighting_keywords = ['lighting', 'golden hour', 'natural light', 'studio light', 'dramatic', 'volumetric']
    style_keywords = ['photorealistic', 'documentary', 'cinematic', 'raw photography', 'realistic']
    composition_keywords = ['composition', 'depth of field', 'bokeh', 'focus', 'rule of thirds']
    
    # Check what's missing
    has_quality = any(kw.lower() in prompt.lower() for kw in quality_keywords)
    has_camera = any(kw.lower() in prompt.lower() for kw in camera_keywords)
    has_lighting = any(kw.lower() in prompt.lower() for kw in lighting_keywords)
    has_style = any(kw.lower() in prompt.lower() for kw in style_keywords)
    has_composition = any(kw.lower() in prompt.lower() for kw in composition_keywords)
    
    # Build enhancement suffix
    enhancements = []
    
    if not has_quality:
        enhancements.append("8K resolution")
        enhancements.append("ultra high detail")
        enhancements.append("professional photography")
    
    if not has_camera:
        # Choose camera based on content tone
        if polarity == "hate":
            enhancements.append("shot on Nikon Z9 with 35mm f/1.8 lens")
        else:
            enhancements.append("shot on Canon EOS R5 with 85mm f/1.4 lens")
    
    if not has_lighting:
        if polarity == "hate":
            enhancements.append("dramatic harsh lighting")
        else:
            enhancements.append("natural golden hour lighting")
    
    if not has_style:
        enhancements.append("photorealistic documentary style")
    
    if not has_composition:
        enhancements.append("cinematic composition with sharp focus")
    
    # Add enhancements if needed
    if enhancements:
        # Remove trailing period if present
        if prompt.endswith('.'):
            prompt = prompt[:-1]
        
        # Add enhancements
        enhanced = prompt + ", " + ", ".join(enhancements)
        logger.debug(f"Enhanced prompt with: {', '.join(enhancements)}")
        return enhanced
    
    return prompt


def validate_prompt_quality(prompt: str) -> tuple[bool, str]:
    """
    Validate that a prompt meets production quality standards.
    Returns (is_valid, reason).
    """
    if not prompt or len(prompt) < MINIMUM_PROMPT_LENGTH:
        return False, f"Prompt too short (minimum {MINIMUM_PROMPT_LENGTH} chars)"
    
    # Must have at least some technical specifications
    required_elements = [
        ('resolution/quality', ['8k', '8K', 'high detail', 'ultra', 'HD', 'resolution']),
        ('camera/photography', ['shot on','Sony', 'lens', 'mm', 'photography', 'DSLR']),
        ('lighting', ['lighting', 'light', 'golden hour', 'natural', 'dramatic', 'studio']),
        ('style', ['photorealistic', 'realistic', 'documentary', 'cinematic', 'raw', 'photo']),
    ]
    
    missing = []
    for element_name, keywords in required_elements:
        if not any(kw.lower() in prompt.lower() for kw in keywords):
            missing.append(element_name)
    
    if missing:
        return False, f"Missing elements: {', '.join(missing)}"
    
    return True, "Valid"


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(MAX_RETRIES),
    reraise=True
)
def generate_single_prompt(
    generator: PromptGeneratorEnhanced,
    text: str,
    class_label: str,
    polarity: str,
    idx: int
) -> str:
    """Generate a single prompt with retry logic."""
    prediction = generator.forward(
        text=text,
        class_label=class_label,
        polarity=polarity
    )
    
    prompt = extract_prompt_from_prediction(prediction, idx)
    
    if not prompt or len(prompt) < MINIMUM_GENERATED_LENGTH:
        raise ValueError(f"Generated prompt too short: {len(prompt)} chars")
    
    return prompt


def process_batch(
    generator: PromptGeneratorEnhanced,
    batch_indices: List[int],
    df: pd.DataFrame,
) -> Dict[int, str]:
    """Process a batch of rows with retry logic and improved error handling."""
    results: Dict[int, str] = {}
    
    # Verify all indices exist in DataFrame
    valid_indices = [idx for idx in batch_indices if idx in df.index]
    if len(valid_indices) != len(batch_indices):
        invalid = set(batch_indices) - set(valid_indices)
        logger.warning(f"Skipping {len(invalid)} invalid indices: {list(invalid)[:10]}")
        for idx in invalid:
            results[idx] = ""
    
    if not valid_indices:
        return results
    
    # Process each item with retry logic
    for idx in valid_indices:
        try:
            # Get row data
            text = str(df.loc[idx]["text"])
            class_label = str(df.loc[idx]["class_label"])
            polarity = str(df.loc[idx]["polarity"])
            
            # Generate prompt with retry logic
            prompt = generate_single_prompt(generator, text, class_label, polarity, idx)
            
            # Post-process to ensure quality
            enhanced_prompt = post_process_prompt(prompt, polarity)
            
            # Validate final prompt
            is_valid, reason = validate_prompt_quality(enhanced_prompt)
            
            if is_valid:
                results[idx] = enhanced_prompt
            else:
                logger.warning(f"Row {idx}: Validation failed - {reason}")
                # Still save it
                results[idx] = enhanced_prompt
                
        except Exception as e:
            logger.error(f"Row {idx} failed after {MAX_RETRIES} retries with {type(e).__name__}: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            results[idx] = ""
    
    return results


def flush_prompts_to_csv(df: pd.DataFrame, prompts: Dict[int, str], output_path: str, keep_recent: int = 1000) -> Dict[int, str]:
    """Flush prompts to CSV and keep only recent ones in memory to prevent memory leaks."""
    try:
        # Update CSV with current prompts
        update_output_csv(df, prompts, output_path)
        
        # Keep only the most recent prompts in memory
        if len(prompts) > keep_recent:
            sorted_indices = sorted(prompts.keys(), reverse=True)  # Latest first
            recent_prompts = {idx: prompts[idx] for idx in sorted_indices[:keep_recent]}
            logger.info(f"Memory management: Kept {len(recent_prompts)} recent prompts, flushed {len(prompts) - len(recent_prompts)}")
            return recent_prompts
        
        return prompts
        
    except Exception as e:
        logger.error(f"Failed to flush prompts: {e}")
        return prompts


def main() -> None:
    """Main execution function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded dataset: {len(df)} rows")
    logger.info(f"DataFrame index range: {df.index.min()} to {df.index.max()}")
    
    # Ensure DataFrame has proper 0-based sequential index for mapping
    if not df.index.equals(pd.RangeIndex(len(df))):
        logger.warning("Resetting DataFrame index to 0-based sequential")
        df = df.reset_index(drop=True)
    
    # Verify dataset structure
    if len(df) < 17000 or len(df) > 19000:
        logger.warning(f"Expected ~18k rows, got {len(df)}. Verify dataset integrity.")

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_indices: Set[int] = set(checkpoint["processed_indices"])
    prompts: Dict[int, str] = checkpoint["prompts"]
    current_batch_size: int = checkpoint["batch_size"]
    
    # Memory management: If we have too many prompts in memory, flush older ones
    if len(prompts) > MAX_PROMPTS_IN_MEMORY:
        logger.info(f"Initial memory management: {len(prompts)} prompts in checkpoint")
        prompts = flush_prompts_to_csv(df, prompts, OUTPUT_CSV, MAX_PROMPTS_IN_MEMORY)

    # Get all valid DataFrame indices
    all_indices = df.index.tolist()
    
    # Filter out already processed indices
    remaining_indices = [idx for idx in all_indices if idx not in processed_indices]
    
    if remaining_indices:
        logger.info(f"Resuming: {len(processed_indices)} done, {len(remaining_indices)} remaining")
    else:
        logger.info("All rows already processed!")
        update_output_csv(df, prompts, OUTPUT_CSV)
        
        # Report statistics
        total = len(all_indices)
        filled = sum(1 for idx in all_indices if prompts.get(idx, "").strip() != "")
        empty = total - filled
        
        valid_count = 0
        for idx in all_indices:
            prompt = prompts.get(idx, "")
            if prompt:
                is_valid, _ = validate_prompt_quality(prompt)
                if is_valid:
                    valid_count += 1
        
        logger.info(f"Stats: {filled}/{total} prompts, {valid_count}/{filled} pass validation")
        return
    
    # Suppress verbose LiteLLM logging
    import litellm
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # Initialize language model
    logger.info(f"Connecting to Ollama at {REMOTE_OLLAMA_URL}")
    logger.info(f"Using model: {MODEL_NAME}")
    
    lm = dspy.LM(
        MODEL_NAME,
        provider="litellm",
        api_base=REMOTE_OLLAMA_URL,
        temperature=0.3,
        max_tokens=512,
        num_retries=4,
    )
    dspy.settings.configure(lm=lm)

    generator = PromptGeneratorEnhanced()
    
    logger.info("=" * 60)
    logger.info("STARTING ENHANCED GENERATION")
    logger.info("=" * 60)
    logger.info(f"Batch size: {current_batch_size}")
    logger.info(f"Checkpoint interval: {ROWS_PER_CHECKPOINT} rows")
    logger.info(f"Memory limit: {MAX_PROMPTS_IN_MEMORY} prompts")
    logger.info("All prompts will include professional specifications")
    logger.info("=" * 60)

    # Process in batches
    pbar = tqdm(total=len(remaining_indices), desc="Generating enhanced prompts")
    rows_since_last_checkpoint = 0
    
    i = 0
    while i < len(remaining_indices):
        batch_start = i
        batch_end = min(i + current_batch_size, len(remaining_indices))
        batch_indices = remaining_indices[batch_start:batch_end]
        
        try:
            # Process batch
            batch_results = process_batch(generator, batch_indices, df)
            
            # Update prompts dict
            prompts.update(batch_results)
            
            # Update processed_indices
            for idx, prompt in batch_results.items():
                if prompt.strip():
                    processed_indices.add(idx)
            
            # Update progress
            processed_count = batch_end - batch_start
            pbar.update(processed_count)
            rows_since_last_checkpoint += processed_count
            
            # Update checkpoint data
            checkpoint["processed_indices"] = list(processed_indices)
            checkpoint["prompts"] = prompts
            checkpoint["batch_size"] = current_batch_size
            
            # Save checkpoint periodically and manage memory
            if rows_since_last_checkpoint >= ROWS_PER_CHECKPOINT:
                logger.info(f"Checkpoint: {rows_since_last_checkpoint} rows processed")
                save_checkpoint(checkpoint)
                
                # Memory management: flush prompts if we have too many
                if len(prompts) > MAX_PROMPTS_IN_MEMORY:
                    prompts = flush_prompts_to_csv(df, prompts, OUTPUT_CSV, MAX_PROMPTS_IN_MEMORY)
                    # Update checkpoint with reduced prompts dict
                    checkpoint["prompts"] = prompts
                else:
                    update_output_csv(df, prompts, OUTPUT_CSV)
                
                rows_since_last_checkpoint = 0
            
            # Move to next batch
            i = batch_end
            
        except Exception as e:
            logger.warning(f"Batch failed: {e}")
            logger.warning(f"Reducing batch size from {current_batch_size}")
            
            current_batch_size = max(1, current_batch_size // 2)
            
            if current_batch_size < 1:
                logger.error(f"Cannot process batch at {batch_start}, skipping...")
                for idx in batch_indices:
                    prompts[idx] = ""
                i = batch_end
                current_batch_size = 1
    
    pbar.close()

    # Final save
    logger.info("Saving final results...")
    checkpoint["processed_indices"] = list(processed_indices)
    checkpoint["prompts"] = prompts
    checkpoint["batch_size"] = current_batch_size
    save_checkpoint(checkpoint)
    update_output_csv(df, prompts, OUTPUT_CSV)
    
    # Final statistics
    total = len(all_indices)
    filled = sum(1 for idx in all_indices if prompts.get(idx, "").strip() != "")
    empty = total - filled
    
    valid_count = 0
    sample_size = min(100, filled)
    sample_indices = [idx for idx in all_indices if prompts.get(idx, "").strip()][:sample_size]
    for idx in sample_indices:
        is_valid, _ = validate_prompt_quality(prompts[idx])
        if is_valid:
            valid_count += 1
    
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rows: {total}")
    logger.info(f"Prompts generated: {filled}")
    logger.info(f"Empty/Failed: {empty}")
    logger.info(f"Success rate: {(filled/total)*100:.2f}%")
    logger.info(f"Quality (sample {sample_size}): {valid_count}/{sample_size} valid")
    logger.info(f"Output: {OUTPUT_CSV}")
    logger.info("=" * 60)
    
    # Show samples
    logger.info("\nSAMPLE ENHANCED PROMPTS:")
    for idx in sample_indices[:3]:
        logger.info(f"\nRow {idx}:")
        logger.info(f"  Text: {df.loc[idx, 'text'][:80]}...")
        logger.info(f"  Prompt: {prompts[idx][:200]}...")


if __name__ == "__main__":
    main()