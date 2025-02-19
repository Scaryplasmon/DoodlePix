#!/usr/bin/env python
"""
Script to convert HEX color codes to descriptive color names with smart merging.
"""

import os
import re
import json
import argparse
import logging
from tqdm import tqdm
import colorsys
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hex_to_rgb(hex_code):
    """Convert hex to RGB."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_descriptive_name(rgb):
    """Convert RGB values to a descriptive color name."""
    r, g, b = rgb
    # Convert to HSL for better color description
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    
    # Convert hue to degrees
    h_deg = h * 360
    
    # Determine base color from hue
    # Default to grey if saturation is very low (for white/black/grey colors)
    if s < 0.15:
        base = "grey"
    # Otherwise determine base color from hue
    elif h_deg < 15 or h_deg > 345:
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
    
    # Add modifiers based on lightness and saturation
    modifiers = []
    
    # Lightness modifiers
    if l < 0.2:
        modifiers.append("dark")
    elif l > 0.8:
        modifiers.append("light")
        
    # Saturation modifiers (skip if we already determined it's grey)
    if base != "grey":
        if s < 0.3:
            modifiers.append("pastel")
        elif s > 0.8:
            modifiers.append("vibrant")
    
    return " ".join(modifiers + [base])

def merge_colors(colors):
    """
    Merge similar colors intelligently, keeping the most descriptive variant.
    Returns a list of unique merged colors.
    """
    # Split colors into base and modifiers
    color_info = {}
    for color in colors:
        parts = color.split()
        base = parts[-1]
        modifiers = set(parts[:-1])
        
        if base not in color_info:
            color_info[base] = modifiers
        else:
            # Merge modifiers, prioritizing more descriptive ones
            existing_mods = color_info[base]
            # Prioritize certain modifiers
            priority_mods = {'vibrant', 'dark', 'light', 'pastel'}
            new_mods = existing_mods.union(modifiers).intersection(priority_mods)
            if new_mods:
                color_info[base] = new_mods
    
    # Handle compound colors (e.g., "grey green")
    merged = []
    skip_bases = set()
    
    for base, modifiers in color_info.items():
        if base in skip_bases:
            continue
            
        # Check for compound color patterns
        compound_match = None
        for other_base in color_info:
            if f"{base} {other_base}" in colors:
                compound_match = f"{base} {other_base}"
                skip_bases.add(other_base)
                break
        
        if compound_match:
            merged.append(compound_match)
        else:
            color = " ".join(sorted(modifiers) + [base])
            merged.append(color)
    
    return merged

def process_text(text):
    """Replace hex codes with descriptive names in text and merge duplicates intelligently."""
    # Match 6-digit hex codes with # prefix
    hex_pattern = re.compile(r'#[0-9a-fA-F]{6}')
    
    # Find all hex codes and convert to descriptive names
    hex_codes = hex_pattern.findall(text)
    
    # If no hex codes found, return original text and info
    if not hex_codes:
        return text, {"status": "no_hex_codes", "original": text, "colors": []}
    
    color_names = [rgb_to_descriptive_name(hex_to_rgb(hex_code)) for hex_code in hex_codes]
    
    # Separate background color if present
    background_color = None
    if "background" in text:
        for i, hex_code in enumerate(hex_codes):
            if hex_code in text.split("background")[0]:
                continue
            background_color = color_names[i]
            color_names.pop(i)
            hex_codes.pop(i)
            break
    
    # Merge similar colors
    merged_colors = merge_colors(color_names)
    
    # Create the color section
    merged_colors_text = ", ".join(merged_colors) if merged_colors else "grey"
    
    # Add background color if present
    if background_color:
        merged_colors_text += f", {background_color} background"
    elif "background" in text:
        merged_colors_text += ", grey background"
    
    # Get the text parts before and after the color section
    parts = re.split(r'#[0-9a-fA-F]{6}(?:,\s*#[0-9a-fA-F]{6})*(?:\s*background)?', text)
    
    # Combine parts with the new color section
    new_text = parts[0] + merged_colors_text + "".join(parts[1:])
    
    # Clean up multiple commas and spaces
    new_text = re.sub(r',\s*,', ',', new_text)
    new_text = re.sub(r'\s+', ' ', new_text)
    new_text = re.sub(r',\s*$', '', new_text)
    
    return new_text.strip(), {
        "status": "success",
        "original": text,
        "original_colors": color_names,
        "merged_colors": merged_colors,
        "background": background_color
    }

def process_files(input_dir, output_dir):
    """Process all text files in directory and generate conversion log."""
    os.makedirs(output_dir, exist_ok=True)
    
    conversion_log = {
        "total_files": 0,
        "successful": 0,
        "no_hex_codes": 0,
        "errors": 0,
        "file_details": {}
    }
    
    # Find all .txt files
    txt_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    conversion_log["total_files"] = len(txt_files)
    logger.info(f"Found {len(txt_files)} text files")
    
    # Process each file
    for input_path in tqdm(txt_files, desc="Processing files"):
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process content
            new_content, process_info = process_text(content)
            
            # Update conversion log
            rel_path = os.path.relpath(input_path, input_dir)
            conversion_log["file_details"][rel_path] = process_info
            
            if process_info["status"] == "success":
                conversion_log["successful"] += 1
            elif process_info["status"] == "no_hex_codes":
                conversion_log["no_hex_codes"] += 1
            
            # Create output path preserving directory structure
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            conversion_log["errors"] += 1
            conversion_log["file_details"][rel_path] = {
                "status": "error",
                "error": str(e)
            }
    
    # Save conversion log
    log_path = os.path.join(output_dir, "conversion_log.json")
    with open(log_path, 'w') as f:
        json.dump(conversion_log, f, indent=2)
    
    logger.info(f"Conversion complete. Log saved to {log_path}")
    logger.info(f"Successfully processed: {conversion_log['successful']}")
    logger.info(f"Files with no hex codes: {conversion_log['no_hex_codes']}")
    logger.info(f"Errors: {conversion_log['errors']}")

def main():
    parser = argparse.ArgumentParser(description="Convert hex color codes to descriptive names in text files")
    parser.add_argument("--input_dir", "-i", required=True, help="Input directory containing text files")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for processed files")
    args = parser.parse_args()
    
    process_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()