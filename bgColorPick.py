#!/usr/bin/env python
"""
Script to fix background colors in prompt files by analyzing the actual images.
"""

import os
import re
import argparse
import logging
import colorsys
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rgb_to_descriptive_name(rgb):
    """Convert RGB values to a descriptive color name."""
    r, g, b = rgb
    
    # Special case for black (very dark colors)
    if max(r, g, b) < 30:
        return "black"
    
    # Special case for white (very light colors)
    if min(r, g, b) > 225:
        return "white"
    
    # Special case for true greys (r≈g≈b)
    if abs(r - g) < 10 and abs(g - b) < 10 and abs(r - b) < 10:
        if r < 80:
            return "dark grey"
        elif r > 180:
            return "light grey"
        else:
            return "grey"
    
    # Convert to HSL for better color description
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    
    # Convert hue to degrees
    h_deg = h * 360
    
    # Determine base color from hue
    if h_deg < 15 or h_deg > 345:
        base = "red"
    elif 15 <= h_deg < 45:
        base = "orange"
    elif 45 <= h_deg < 75:
        base = "yellow"
    elif 75 <= h_deg < 165:
        base = "green"
    elif 165 <= h_deg < 195:
        base = "cyan"
    elif 195 <= h_deg < 255:
        base = "blue"
    elif 255 <= h_deg < 285:
        base = "purple"
    else:  # 285 <= h_deg <= 345
        base = "pink"
    
    # Add at most one modifier based on lightness and saturation
    if l < 0.2:
        return f"dark {base}"
    elif l > 0.8:
        return f"light {base}"
    elif s < 0.3 and base != "grey":
        return f"pastel {base}"
    elif s > 0.8:
        return f"vibrant {base}"
    else:
        return base

def get_background_color_from_image(image_path):
    """
    Extract background color by sampling pixels near the corners of the image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Sample points near corners (8px and 16px from corners)
        sample_points = [
            # 8px from corners
            (8, 8), (width-8, 8), (8, height-8), (width-8, height-8),
            # 16px from corners
            (16, 16), (width-16, 16), (16, height-16), (width-16, height-16)
        ]
        
        # Get RGB values for sample points
        samples = []
        for x, y in sample_points:
            samples.append(img.getpixel((x, y)))
        
        # Calculate mean color
        mean_color = np.mean(samples, axis=0).astype(int)
        
        # Convert to descriptive name
        color_name = rgb_to_descriptive_name(mean_color)
        
        return color_name
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return "grey"  # Default fallback

def fix_background_colors(prompt_dir, image_dir, output_dir):
    """
    Find prompts with 'grey background', analyze matching images, and fix the background color.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all text files with "grey background"
    grey_bg_files = []
    for root, _, files in os.walk(prompt_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "grey background." in content:
                        grey_bg_files.append((file_path, content))
    
    logger.info(f"Found {len(grey_bg_files)} files with 'grey background'")
    
    # Process each file
    fixed_count = 0
    error_count = 0
    
    for file_path, content in tqdm(grey_bg_files, desc="Fixing background colors"):
        try:
            # Get base filename without extension
            base_name = os.path.basename(file_path)
            file_name_no_ext = os.path.splitext(base_name)[0]
            
            # Find matching image file
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(image_dir, file_name_no_ext + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if not image_path:
                logger.warning(f"No matching image found for {file_path}")
                continue
            
            # Get background color from image
            bg_color = get_background_color_from_image(image_path)
            
            # Replace "grey background" with the correct color
            fixed_content = content.replace("grey background.", f"{bg_color} background.")
            
            # Create output path
            rel_path = os.path.relpath(file_path, prompt_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write fixed content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
                
            fixed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            error_count += 1
    
    logger.info(f"Background color fixing complete.")
    logger.info(f"Fixed: {fixed_count}")
    logger.info(f"Errors: {error_count}")

def main():
    parser = argparse.ArgumentParser(description="Fix background colors in prompt files by analyzing images")
    parser.add_argument("--prompt_dir", "-p", required=True, help="Directory containing prompt text files")
    parser.add_argument("--image_dir", "-i", required=True, help="Directory containing corresponding images")
    parser.add_argument("--output_dir", "-o", required=False, help="Output directory for fixed prompt files")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.prompt_dir + "_fixed"
    
    fix_background_colors(args.prompt_dir, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()