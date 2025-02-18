import os
import shutil
from pathlib import Path

def copy_matching_txt_files(input_folder, txt_source_folder):
    """
    Copy txt files from source folder to input folder if they match image names.
    
    Args:
        input_folder: Folder containing the images
        txt_source_folder: Folder containing the txt files to copy from
    """
    # Create input folder if it doesn't exist
    Path(input_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all image files from input folder
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    image_files = [f.stem for f in Path(input_folder).glob('*') if f.suffix.lower() in img_extensions]
    
    # Counter for copied files
    copied_count = 0
    not_found_count = 0
    
    print(f"Found {len(image_files)} images in input folder")
    
    # Iterate through image names and look for matching txt files
    for img_name in image_files:
        txt_path = Path(txt_source_folder) / f"{img_name}.txt"
        
        if txt_path.exists():
            try:
                # Copy the txt file to input folder
                shutil.copy2(txt_path, Path(input_folder) / txt_path.name)
                copied_count += 1
                print(f"Copied: {txt_path.name}")
            except Exception as e:
                print(f"Error copying {txt_path.name}: {str(e)}")
        else:
            not_found_count += 1
            print(f"No matching txt file found for: {img_name}")
    
    print(f"\nSummary:")
    print(f"Total txt files copied: {copied_count}")
    print(f"Images without matching txt files: {not_found_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Copy matching txt files to image folder')
    parser.add_argument('--input_folder', '-i', type=str, required=True,
                      help='Folder containing the images')
    parser.add_argument('--txt_source_folder', '-t', type=str, required=True,
                      help='Folder containing the source txt files')
    """
        python selectTxt_imgBased.py -i "DoodlePixV4/DoodlePixV5/edited_image/Doodles/" -t "DoodlePixV4/DoodlePixV5/edited_image/fidelity/"
    """
    args = parser.parse_args()
    copy_matching_txt_files(args.input_folder, args.txt_source_folder)