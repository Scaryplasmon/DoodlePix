import os
from pathlib import Path
from tqdm import tqdm
import re

def read_and_clean_file(file_path: str) -> str:
    """Read file content and replace newlines with spaces."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('\n', ', ').strip()
            content = content.replace('background,', 'background.')
        return content
    except Exception as e:
        print(f"\nError reading file {file_path}: {str(e)}")
        return ""

def get_existing_content(file_path: str) -> tuple:
    """Read existing content and split into components."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Split content into components
        parts = content.split(',', 2)  # Split at first two commas
        
        fidelity = parts[0].strip() if len(parts) > 0 else ""
        tags = parts[1].strip() if len(parts) > 1 else ""
        colors = parts[2].strip() if len(parts) > 2 else ""
        
        return fidelity, tags, colors
    except FileNotFoundError:
        return "", "", ""
    except Exception as e:
        print(f"\nError reading existing content from {file_path}: {str(e)}")
        return "", "", ""

def merge_txt_files(base_path: str, process_flags: dict):
    """
    Merge txt files based on specified flags.
    process_flags is a dict with keys 'fidelity', 'tags', 'colors' and boolean values.
    """
    # Define folders to process based on flags
    folders = []
    if process_flags['all'] or process_flags['fidelity']:
        folders.append('fidelity')
    if process_flags['all'] or process_flags['tags']:
        folders.append('tags')
    if process_flags['all'] or process_flags['colors']:
        folders.append('colors')
    
    if not folders:
        print("No folders selected to process!")
        return
    
    # Create output folder
    output_folder = os.path.join(base_path, 'edit_prompt')
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of txt files from first folder
    first_folder = os.path.join(base_path, folders[0])
    txt_files = [f for f in os.listdir(first_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print("No txt files found in the first folder!")
        return
    
    print(f"Found {len(txt_files)} txt files to process")
    print(f"Processing folders: {', '.join(folders)}")
    
    # Process each txt file
    for txt_file in tqdm(txt_files, desc="Processing files"):
        try:
            # Get existing content if any
            existing_fidelity, existing_tags, existing_colors = get_existing_content(
                os.path.join(output_folder, txt_file)
            )
            
            # Read new content from selected folders
            new_contents = {
                'fidelity': existing_fidelity,
                'tags': existing_tags,
                'colors': existing_colors
            }
            
            for folder in folders:
                file_path = os.path.join(base_path, folder, txt_file)
                if not os.path.exists(file_path):
                    print(f"\nWarning: Missing file in {folder}: {txt_file}")
                    continue
                content = read_and_clean_file(file_path)
                
                if folder == 'tags':
                    content = f"<tags: {content}>"
                new_contents[folder] = content
            
            # Merge contents
            merged_content = f"{new_contents['fidelity']}, {new_contents['tags']}, {new_contents['colors']}"
            
            # Save merged content
            output_path = os.path.join(output_folder, txt_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(merged_content)
                
        except Exception as e:
            print(f"\nError processing {txt_file}: {str(e)}")
            continue
    
    print("\nProcessing completed!")
    print(f"Merged files saved in: {output_folder}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge txt files from multiple folders')
    parser.add_argument('--input_path', '-i', type=str, required=True,
                       help='Base path containing the fidelity, tags, and colors folders')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Process all folders')
    parser.add_argument('--fidelity', '-f', action='store_true',
                       help='Process fidelity folder')
    parser.add_argument('--tags', '-t', action='store_true',
                       help='Process tags folder')
    parser.add_argument('--colors', '-c', action='store_true',
                       help='Process colors folder')
    
    args = parser.parse_args()
    
    # Create process flags dictionary
    process_flags = {
        'all': args.all,
        'fidelity': args.fidelity,
        'tags': args.tags,
        'colors': args.colors
    }
    
    merge_txt_files(args.input_path, process_flags)
    
    """
    Example usage:
    # Process all folders:
    python joinPrompts.py -i path/to/folder -a
    
    # Process only specific folders:
    python joinPrompts.py -i path/to/folder -f -t  # Process fidelity and tags only
    python joinPrompts.py -i path/to/folder -c     # Process only colors
    """