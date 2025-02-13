from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import colorsys
from collections import defaultdict
#4.41.2
# revision = "2024-08-26"  # Pin to specific version
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, trust_remote_code=True, revision=revision,
#     torch_dtype=torch.float16, attn_implementation="eager"
# ).to("cuda")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)

def get_color_name(rgb):
    """Convert RGB to closest basic color name with tone indication"""
    # Dictionary of basic colors and their RGB values
    colors = {
        'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
        'yellow': (255, 255, 0), 'purple': (128, 0, 128), 'orange': (255, 165, 0),
        'brown': (165, 42, 42), 'pink': (255, 192, 203), 'gray': (128, 128, 128),
        'black': (0, 0, 0), 'white': (255, 255, 255)
    }
    
    # Convert RGB to HSV for better color comparison
    h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
    
    # Add tone indicators
    tone = ""
    if v < 0.3: tone = "dark"
    elif v > 0.7: tone = "light"
    elif s < 0.3: tone = "grayish"
    elif s > 0.7: tone = "vibrant"
    
    # Find closest basic color
    min_dist = float('inf')
    closest_color = None
    for color_name, color_rgb in colors.items():
        dist = sum((a-b)**2 for a, b in zip(rgb, color_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
    
    return f"{tone} {closest_color}".strip() if tone else closest_color

def analyze_image_colors(image, n_colors=3):
    """Extract dominant colors and their locations in the image"""
    # Convert image to numpy array
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    
    # Use K-means to find dominant colors with explicit n_init
    kmeans = KMeans(
        n_clusters=n_colors, 
        random_state=42,
        n_init=10  # Explicitly set n_init to remove warning
    )
    kmeans.fit(pixels)
    
    # Get the dominant colors and their percentages
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    percentages = np.bincount(labels) / len(labels)
    
    # Convert to color names and percentages
    color_info = []
    for color, percentage in zip(colors, percentages):
        color_name = get_color_name(tuple(color))
        hex_code = '#{:02x}{:02x}{:02x}'.format(*color)
        color_info.append({
            'name': color_name,
            'hex': hex_code,
            'percentage': float(percentage)
        })
    
    return sorted(color_info, key=lambda x: x['percentage'], reverse=True)

def analyze_style(image):
    """
    Analyze image style based on visual characteristics using Moondream
    Returns: style type (flat, 3d, or painted)
    """
    # Ask specific questions about visual characteristics
    style_prompt = (
        "Analyze this icon's visual style. "
        "Consider these aspects and answer with ONLY ONE of these terms - flat, 3d, or painted: "
        "1. If it has low contrast and solid shadow shapes, it's flat. "
        "2. If it has high contrast and realistically blended shadows on surfaces, it's 3d. "
        "3. If it has vibrant colors, mixed intensity shadows, and glossy appearance, it's painted."
    )
    
    style_response = model.query(image, style_prompt)["answer"].lower().strip()
    
    # Ensure we get one of our desired styles
    valid_styles = {'flat', '3d', 'painted'}
    return style_response if style_response in valid_styles else 'flat'

def analyze_detail_level(image):
    """Get detail level using Moondream's analysis"""
    detail_prompt = (
        "Rate the level of detail in this icon from 1-10, considering: "
        "1. Amount of decorative elements, the more the higher the detail level"
        "2. Complexity of shading "
        "3. Number of distinct parts "
        "4. Intricacy of lines and patterns "
        "Answer with ONLY a number from 1-10."

    )
    
    try:
        detail_response = model.query(image, detail_prompt)["answer"]
        # Extract number from response
        detail_level = int(''.join(filter(str.isdigit, detail_response)))
        return max(1, min(10, detail_level))  # Ensure value is between 1-10
    except:
        return 5  # Default fallback value

def parse_existing_description(txt_content):
    attributes = {
        'subject': None,
        'style': None,
        'colors': None,
        'materials': None,
        'theme': None,
        'details': None,
        'description': None
    }
    
    # Parse existing content
    for attr in attributes.keys():
        search_pattern = f"<{attr}: "
        if search_pattern in txt_content:
            start_idx = txt_content.find(search_pattern) + len(search_pattern)
            end_idx = txt_content.find('>', start_idx)
            if end_idx != -1:
                attributes[attr] = txt_content[start_idx:end_idx]
    
    return attributes

def get_color_material_description(image, color_info):
    """Get a concise description of colors and their associated materials"""
    try:
        # Create a specific prompt for color-material association
        color_prompt = (
            "What are the main materials and their colors in this icon? "
            "Answer in this format only: material1 (color1), material2 (color2)"
        )
        
        materials_colors = model.query(image, color_prompt)["answer"].lower()
        return materials_colors
    except Exception as e:
        print(f"Color-material analysis error: {str(e)}")
        return color_info[0]['name'] if color_info else "unknown"

def analyze_world_theme(image):
    """Determine which world/theme the icon belongs to"""
    world_prompt = """
    Which world does this icon belong to? Choose which world describes the icon best among the following options:
    1. WHIMSICAL: A cute and playful world with:
    - Adorably proportioned characters and objects
    - Soft, rounded shapes and pastel colors
    - Cheerful, innocent aesthetic
    - Playful, toy-like elements
    
    2. FANTASY: A mystical realm of magic, forests and castles, filled with:
    - Ancient dungeons and magical artifacts
    - Wizards, spells, and enchanted items
    - Medieval fantasy elements like scrolls, potions, weapons
    - Magical creatures and ethereal effects
        
    3. SCI-FI: An advanced technological future with:
    - High-tech devices and futuristic gear
    - Energy-based technology
    - Sleek, modern designs
    - Advanced scientific elements

    4. STEAMPUNK: A post-apocalyptic desert world featuring:
    - Industrial machinery and brass, scraps or bronze mechanisms
    - Desert wasteland elements
    - Rusty metals and worn materials
    - Steam-powered technology
    

    Answer with the option that best describes the icon.
    """
    

    world = model.query(image, world_prompt)["answer"].upper().strip()
    valid_worlds = {'FANTASY', 'WHIMSICAL', 'STEAMPUNK', 'SCI-FI'}
    return world.lower() if world in valid_worlds else 'fantasy'

def get_structured_description(image, existing_attributes=None):
    if existing_attributes is None:
        existing_attributes = {}
    
    tags = model.query(
        image, 
        "Give this image a minimum of 2 and maximum of 6 tags, in order of importance, separated by commas. Never repeat the same tag twice, you can add synonyms. Describe the subject of the image with tags. What is this icon?"
    )["answer"].lower()

    # Split tags by comma and clean up whitespace
    tag_list = [tag.strip() for tag in tags.split(',')]
    
    # Remove empty tags and unwanted style tags
    unwanted_tags = {'flat', '3d', 'painted', '3d renderer', '3d rendering', "outline", "black outline", "drawing"}
    tag_list = [tag for tag in tag_list if tag and tag not in unwanted_tags]
    
    # Remove duplicate tags while preserving order
    seen = set()
    unique_tag_list = []
    for tag in tag_list:
        if tag not in seen:
            seen.add(tag)
            unique_tag_list.append(tag)
    tag_list = unique_tag_list
    
    # Ensure minimum of 2 tags
    if len(tag_list) < 2:
        print(f"Warning: Only {len(tag_list)} tags found. Adding generic tag.")
        tag_list.append("icon")
    
    # Limit to maximum 5 tags
    tag_list = tag_list[:6]
    
    # Join tags back with commas
    cleaned_tags = ', '.join(tag_list)

    prompt = (
        f"<tags: {cleaned_tags}>"
    )
    
    return prompt

def process_folder(folder_path, override=False):
    """
    Process all images in a folder
    Args:
        folder_path: Path to folder containing images
        override: If True, will override existing txt files. If False, will skip them.
    """
    results = {}
    
    # Get all PNG files in the folder
    image_files = list(Path(folder_path).glob('*.png'))
    
    for img_path in image_files:
        try:
            # Check if corresponding txt file exists
            txt_path = img_path.with_suffix('.txt')
            existing_attributes = {}
            
            if txt_path.exists() and not override:
                print(f"Skipping {img_path.name} - txt file exists. Use --override to force processing")
                continue
            elif txt_path.exists() and override:
                print(f"Overriding existing file for {img_path.name}")
            else:
                print(f"Processing new file: {img_path.name}")
            
            # Load and process image
            image = Image.open(img_path)
            
            # Get structured description
            description = get_structured_description(image)
            
            # Store result using filename as key
            results[img_path.name] = description
            
            # Save individual txt file
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(description)
            
            print(f"Description: {description}")
            print(f"Saved to: {txt_path.name}\n")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
    
    # Save all results to json
    output_path = os.path.join(folder_path, 'image_descriptions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll results saved to: {output_path}")
    print(f"Individual .txt files saved alongside each image")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--folder_path', "-i", type=str, default="./DoodlePixV4/edited_image/",
                      help='Path to folder containing images')
    parser.add_argument('--override', action='store_true',
                      help='Override existing txt files')
    
    args = parser.parse_args()
    process_folder(args.folder_path, args.override)