import os
import re
import random
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import concurrent.futures

def process_dataset(
    input_img_dir,
    output_img_dir, 
    txt_dir,
    sidebyside_dir,
    fidelity_txt_dir,
    question_dir,
    samples_per_higher_fidelity=100
):
    """
    Process dataset according to specifications.
    
    Args:
        input_img_dir: Directory containing input images
        output_img_dir: Directory containing output images
        txt_dir: Directory containing text files
        sidebyside_dir: Directory to save merged images
        fidelity_txt_dir: Directory to save fidelity text files
        question_dir: Directory to save question text files
        samples_per_higher_fidelity: Number of samples to select for fidelity values 5-9
    """
    # Create output directories if they don't exist
    for directory in [sidebyside_dir, fidelity_txt_dir, question_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print("Scanning text files and extracting fidelity values...")
    
    # Dictionary to store files by fidelity value
    fidelity_files = {i: [] for i in range(10)}
    
    # Scan all text files to extract fidelity values
    for filename in os.listdir(txt_dir):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(txt_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Extract fidelity value (f followed by a digit)
            match = re.search(r'f(\d)', content)
            if match:
                fidelity = int(match.group(1))
                if 0 <= fidelity <= 9:  # Only consider fidelity values 0-9
                    fidelity_files[fidelity].append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Select files based on fidelity criteria
    selected_files = []
    
    # For fidelity 0-4, select all files
    for fidelity in range(5):
        selected_files.extend(fidelity_files[fidelity])
        print(f"Selected {len(fidelity_files[fidelity])} files with fidelity {fidelity}")
    
    # For fidelity 5-9, select up to 100 files each
    for fidelity in range(5, 10):
        if len(fidelity_files[fidelity]) <= samples_per_higher_fidelity:
            selected_files.extend(fidelity_files[fidelity])
            print(f"Selected all {len(fidelity_files[fidelity])} files with fidelity {fidelity}")
        else:
            sampled = random.sample(fidelity_files[fidelity], samples_per_higher_fidelity)
            selected_files.extend(sampled)
            print(f"Selected {samples_per_higher_fidelity} files out of {len(fidelity_files[fidelity])} with fidelity {fidelity}")
    
    print(f"Total selected files: {len(selected_files)}")
    
    # Process each selected file
    def process_file(filename):
        basename = os.path.splitext(filename)[0]
        
        # Check if corresponding images exist
        input_img_path = os.path.join(input_img_dir, basename + '.png')  # Adjust extension if needed
        output_img_path = os.path.join(output_img_dir, basename + '.png')  # Adjust extension if needed
        
        if not os.path.exists(input_img_path) or not os.path.exists(output_img_path):
            return f"Missing image for {basename}"
        
        # Extract fidelity from text file
        txt_path = os.path.join(txt_dir, filename)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            match = re.search(r'(f\d)', content)
            if not match:
                return f"Could not extract fidelity from {filename}"
                
            fidelity_value = match.group(1)
            
            # Create side-by-side image
            img1 = Image.open(input_img_path)
            img2 = Image.open(output_img_path)
            
            # Ensure both images are 512x512
            if img1.size != (512, 512) or img2.size != (512, 512):
                img1 = img1.resize((512, 512))
                img2 = img2.resize((512, 512))
            
            # Create new 1024x512 image
            merged_img = Image.new('RGB', (1024, 512))
            merged_img.paste(img1, (0, 0))
            merged_img.paste(img2, (512, 0))
            
            # Save merged image
            merged_img_path = os.path.join(sidebyside_dir, basename + '.png')
            merged_img.save(merged_img_path)
            
            # Save fidelity text file
            fidelity_txt_path = os.path.join(fidelity_txt_dir, basename + '.txt')
            with open(fidelity_txt_path, 'w', encoding='utf-8') as f:
                f.write(fidelity_value)
            
            # Save question text file
            question_txt_path = os.path.join(question_dir, basename + '.txt')
            with open(question_txt_path, 'w', encoding='utf-8') as f:
                f.write("what is the fidelity value of this drawing?")
                
            return f"Successfully processed {basename}"
        except Exception as e:
            return f"Error processing {basename}: {e}"
    
    # Process files in parallel
    print("Processing selected files...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_file, selected_files), total=len(selected_files)))
    
    # Count success and failures
    successes = [r for r in results if r.startswith("Successfully")]
    failures = [r for r in results if not r.startswith("Successfully")]
    
    print(f"Processing complete. {len(successes)} files processed successfully, {len(failures)} failures.")
    if failures:
        print("First 10 failures:")
        for failure in failures[:10]:
            print(f"  {failure}")
    
    return {
        "total_selected": len(selected_files),
        "successful_processed": len(successes),
        "failures": len(failures)
    }

if __name__ == "__main__":
    # Set your directory paths here
    base_dir = "data/DoodlePixV6/"  # Replace with your dataset path
    input_img_dir = os.path.join(base_dir, "input_image")
    output_img_dir = os.path.join(base_dir, "edited_image")
    txt_dir = os.path.join(base_dir, "edit_prompt")
    
    # Output directories
    sidebyside_dir = os.path.join(base_dir, "sidebysideImgs")
    fidelity_txt_dir = os.path.join(base_dir, "fidelity_txt")
    question_dir = os.path.join(base_dir, "question")
    
    # Process the dataset
    stats = process_dataset(
        input_img_dir=input_img_dir,
        output_img_dir=output_img_dir,
        txt_dir=txt_dir,
        sidebyside_dir=sidebyside_dir,
        fidelity_txt_dir=fidelity_txt_dir,
        question_dir=question_dir,
        samples_per_higher_fidelity=100
    )
    
    print("\nSummary statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")