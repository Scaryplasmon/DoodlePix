import os
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Set
from tqdm import tqdm

def setup_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Copy random images from multiple folders')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Root folder containing subfolders with images')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Destination folder for copied images')
    parser.add_argument('--base_images', type=int, default=50,
                       help='Base number of images to select from first folder')
    parser.add_argument('--increment', type=int, default=25,
                       help='Increment in number of images for each subsequent folder')
    return parser.parse_args()

def get_image_files(folder: str) -> List[str]:
    """Get all image files from a folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    return [f for f in os.listdir(folder) 
            if os.path.splitext(f)[1].lower() in image_extensions]

def create_fidelity_record(output_folder: str, image_name: str, source_folder: str):
    """Create a text file recording the source folder for each selected image."""
    fidelity_folder = os.path.join(output_folder, 'fidelity')
    os.makedirs(fidelity_folder, exist_ok=True)
    
    txt_path = os.path.join(fidelity_folder, f"{os.path.splitext(image_name)[0]}.txt")
    with open(txt_path, 'w') as f:
        f.write(source_folder)

def calculate_folder_distributions(num_folders: int, total_images: int) -> List[int]:
    """
    Calculate how many images to take from each folder using a progressive distribution.
    Returns a list of numbers where each number represents images to take from corresponding folder.
    """
    # Ensure we don't try to select more images than possible
    if total_images <= 0:
        raise ValueError("No images found in folders")
    
    # Calculate distribution weights (1 to num_folders)
    weights = list(range(1, num_folders + 1))
    total_weight = sum(weights)
    
    # Calculate number of images for each folder
    distributions = []
    remaining_images = total_images
    
    for weight in weights[:-1]:  # Process all except last folder
        num_images = int((weight / total_weight) * total_images)
        distributions.append(num_images)
        remaining_images -= num_images
    
    # Add remaining images to last folder to ensure we use exactly total_images
    distributions.append(remaining_images)
    
    return distributions

def main():
    args = setup_argparse()
    
    try:
        # Create output directories
        os.makedirs(args.output_folder, exist_ok=True)
        os.makedirs(os.path.join(args.output_folder, 'fidelity'), exist_ok=True)
        
        # Get all subfolders and sort them
        subfolders = [f.path for f in os.scandir(args.input_folder) if f.is_dir()]
        if not subfolders:
            raise ValueError(f"No subfolders found in {args.input_folder}")
        subfolders.sort()
        
        # Get number of images in first folder (assuming all have same amount)
        first_folder_images = get_image_files(subfolders[0])
        total_images = len(first_folder_images)
        print(f"Found {len(subfolders)} folders with {total_images} images each")
        
        # Calculate distribution of images across folders
        distributions = calculate_folder_distributions(len(subfolders), total_images)
        
        # Keep track of used image names
        used_image_names: Set[str] = set()
        
        # Process each subfolder with progress bar
        for folder, num_images in tqdm(zip(subfolders, distributions), 
                                     total=len(subfolders),
                                     desc="Processing folders"):
            try:
                # Get all image files from the folder
                image_files = get_image_files(folder)
                
                if len(image_files) < num_images:
                    print(f"\nWarning: Folder {os.path.basename(folder)} has fewer images than required")
                    continue
                
                # Filter out already used image names
                available_images = [img for img in image_files 
                                  if img not in used_image_names]
                
                # Select random images
                num_to_select = min(num_images, len(available_images))
                selected_images = random.sample(available_images, num_to_select)
                
                # Process selected images
                folder_name = os.path.basename(folder)
                for image_name in selected_images:
                    # Copy image to output folder
                    src_path = os.path.join(folder, image_name)
                    dst_path = os.path.join(args.output_folder, image_name)
                    shutil.copy2(src_path, dst_path)
                    
                    # Create fidelity record
                    create_fidelity_record(args.output_folder, image_name, folder_name)
                    
                    # Add to used images set
                    used_image_names.add(image_name)
                
                print(f"\nProcessed {folder_name}: Selected {num_to_select} images")
                
            except Exception as e:
                print(f"\nError processing folder {os.path.basename(folder)}: {str(e)}")
                continue
        
        print(f"\nCompleted! Total images processed: {len(used_image_names)}")
        print(f"Distribution across folders: {distributions}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()