import os
import pandas as pd
from PIL import Image
import io
from pathlib import Path
from tqdm import tqdm
import numpy as np

def image_to_bytes(image_path):
    """Convert image to bytes for parquet storage"""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

def read_text_file(text_path):
    """Read prompt from text file"""
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def create_dataset(original_dir, preprocessed_dirs, text_dir, output_path):
    """
    Create parquet dataset from directories
    
    Args:
        original_dir: Directory containing original images
        preprocessed_dirs: Dict of preprocessor name -> directory with preprocessed images
        text_dir: Directory containing text files
        output_path: Path to save parquet file
    """
    data = []
    
    # Get list of all original images
    original_files = [f for f in os.listdir(original_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(original_files):
        base_name = os.path.splitext(img_name)[0]
        
        # Get original image
        orig_path = os.path.join(original_dir, img_name)
        
        # Get text prompt
        text_path = os.path.join(text_dir, f"{base_name}.txt")
        
        # Skip if text file doesn't exist
        if not os.path.exists(text_path):
            continue
            
        try:
            # Read original image and prompt
            original_image = image_to_bytes(orig_path)
            prompt = read_text_file(text_path)
            
            # Read preprocessed images
            preprocessed_images = {}
            for prep_name, prep_dir in preprocessed_dirs.items():
                prep_path = os.path.join(prep_dir, img_name)
                if os.path.exists(prep_path):
                    preprocessed_images[prep_name] = image_to_bytes(prep_path)
                else:
                    print(f"Warning: Missing preprocessed image {prep_path}")
                    continue
            
            # Only add if we have all preprocessed versions
            if len(preprocessed_images) == len(preprocessed_dirs):
                data.append({
                    'original_image': original_image,
                    'prompt': prompt,
                    **preprocessed_images
                })
                
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue
    
    # Create DataFrame and save to parquet
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    print(f"Created dataset with {len(df)} samples")

if __name__ == "__main__":
    # Example usage
    preprocessed_dirs = {
        'canny_image': 'path/to/canny/images',
        'soft_edge_image': 'path/to/soft_edge/images',
        'doodle_image': 'path/to/doodle/images'
    }
    
    create_dataset(
        original_dir='path/to/original/images',
        preprocessed_dirs=preprocessed_dirs,
        text_dir='path/to/text/files',
        output_path='icon_dataset.parquet'
    )