import os
import shutil
from pathlib import Path

def copy_matching_images(input_folder, target_folder):
    # Create input folder if it doesn't exist
    Path(input_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all txt files from input folder
    txt_files = [f.stem for f in Path(input_folder).glob('*.txt')]
    
    # Common image extensions
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    
    # Counter for copied files
    copied_count = 0
    
    # Iterate through all files in target folder
    for file in Path(target_folder).iterdir():
        if file.suffix.lower() in img_extensions and file.stem in txt_files:
            # Copy the image to input folder
            try:
                shutil.copy2(file, Path(input_folder) / file.name)
                copied_count += 1
                print(f"Copied: {file.name}")
            except Exception as e:
                print(f"Error copying {file.name}: {str(e)}")
    
    print(f"\nTotal files copied: {copied_count}")

if __name__ == "__main__":
    # Replace these paths with your actual folder paths
    input_folder = "DoodlePixV4/DoodlePixV5/A_flat/_bis/bb/"  # folder containing .txt files
    target_folder = "DoodlePixV4/DoodlePixV5/A_flat/edges_A_flat/"  # folder containing images
    
    copy_matching_images(input_folder, target_folder)