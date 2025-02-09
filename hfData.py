import os
import shutil
import json
from pathlib import Path

def create_hf_dataset(source_dir, output_dir):
    """Convert local dataset to HuggingFace ImageFolder format."""
    
    # Create output directory structure
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Get all image files
    input_dir = os.path.join(source_dir, "input_image")
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    metadata = []
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        
        # Get paths
        input_img_path = os.path.join(source_dir, "input_image", img_file)
        edited_img_path = os.path.join(source_dir, "edited_image", img_file)
        prompt_path = os.path.join(source_dir, "edit_prompt", f"{base_name}.txt")
        
        # Read prompt text from file
        with open(prompt_path, 'r', encoding='utf-8') as f:
            edit_prompt = f.read().strip()
        
        # Copy images to train directory
        shutil.copy2(input_img_path, os.path.join(train_dir, img_file))
        shutil.copy2(edited_img_path, os.path.join(train_dir, f"edited_{img_file}"))
        
        # Create metadata entry with file_name field and our three columns
        metadata.append({
            "file_name": img_file,  # Required by ImageFolder
            "input_image": img_file,
            "edited_image": f"edited_{img_file}",
            "edit_prompt": edit_prompt
        })
    
    # Save metadata.jsonl
    with open(os.path.join(train_dir, "metadata.jsonl"), 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    source_directory = "train"
    output_directory = "hf_dataset"
    create_hf_dataset(source_directory, output_directory)
    print(f"Dataset converted and saved to {output_directory}")