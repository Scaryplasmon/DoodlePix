from pathlib import Path

def process_matching_files(image_folder, text_folder):
    # Convert string paths to Path objects if needed
    image_path = Path(image_folder)
    text_path = Path(text_folder)
    
    # Get all PNG files from image folder
    png_files = list(image_path.glob('*.png'))
    
    # Process each PNG file
    for png_file in png_files:
        # Construct the corresponding txt file path
        txt_file = text_path / f"{png_file.stem}.txt"
        
        # Check if matching text file exists
        if txt_file.exists():
            # Read the content
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add the doodle prefix and write back
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"<doodle>, {content}")
            
            print(f"Processed: {txt_file.name}")
        else:
            print(f"No matching text file for: {png_file.name}")

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual folders
    image_folder = "DoodlePix_chest/Doodles/"
    text_folder = "DoodlePixV2/edit_prompt/"
    
    process_matching_files(image_folder, text_folder)