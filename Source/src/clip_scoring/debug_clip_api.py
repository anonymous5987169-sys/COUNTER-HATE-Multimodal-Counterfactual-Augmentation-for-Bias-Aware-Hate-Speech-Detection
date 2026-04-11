"""
Debug script to understand CLIP API output types.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Load model
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Create dummy inputs
dummy_image = Image.new("RGB", (224, 224))
dummy_text = "a photo of a cat"

with torch.no_grad():
    inputs = processor(
        text=[dummy_text],
        images=[dummy_image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    
    # Move to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to("cuda")
    
    # Test different approaches
    print("\n1. Testing full model forward:")
    outputs = model(**inputs)
    print(f"   Type: {type(outputs)}")
    print(f"   Attributes: {dir(outputs)}")
    
    print("\n2. Testing get_image_features:")
    image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
    print(f"   Type: {type(image_features)}")
    print(f"   Shape: {image_features.shape if hasattr(image_features, 'shape') else 'N/A'}")
    if hasattr(image_features, '__dict__'):
        print(f"   Attributes: {image_features.__dict__.keys()}")
    
    print("\n3. Testing get_text_features:")
    text_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    print(f"   Type: {type(text_features)}")
    print(f"   Shape: {text_features.shape if hasattr(text_features, 'shape') else 'N/A'}")
    if hasattr(text_features, '__dict__'):
        print(f"   Attributes: {text_features.__dict__.keys()}")
    
    # Try to extract if it's an object
    print("\n4. Try accessing attributes if available:")
    if hasattr(image_features, 'last_hidden_state'):
        print("   image_features.last_hidden_state exists")
        print(f"   Shape: {image_features.last_hidden_state.shape}")
    if hasattr(image_features, 'pooler_output'):
        print("   image_features.pooler_output exists")
        print(f"   Shape: {image_features.pooler_output.shape}")
