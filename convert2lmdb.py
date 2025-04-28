import os
import io
import pickle
import lmdb
import logging
from PIL import Image as PILImage
from tqdm import tqdm
import re

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_dataset_size(source_dir):
    """Calculate the total size of the dataset and add safety margin."""
    total_size = 0
    input_dir = os.path.join(source_dir, "input_image")
    edited_dir = os.path.join(source_dir, "edited_image")
    prompt_dir = os.path.join(source_dir, "edit_prompt")
    
    # Get size of all files
    for directory in [input_dir, edited_dir, prompt_dir]:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    # Add 50% overhead for LMDB metadata and safety margin
    total_size = int(total_size * 1.5)
    
    # Ensure minimum size of 128MB
    return max(total_size, 128 * 1024 * 1024)

def create_lmdb_dataset(source_dir, lmdb_path, map_size_gb=None):
    """
    Converts an image dataset with input_image, edited_image, and edit_prompt
    subdirectories into a single LMDB database.

    Args:
        source_dir (str): Path to the root directory containing the subdirectories.
        lmdb_path (str): Path where the output LMDB file will be created.
        map_size_gb (float, optional): Override the automatic map size calculation.
    """
    logger.info(f"Starting LMDB creation from: {source_dir}")
    logger.info(f"Output LMDB path: {lmdb_path}")

    # Verify directories exist
    input_dir = os.path.join(source_dir, "input_image")
    edited_dir = os.path.join(source_dir, "edited_image")
    prompt_dir = os.path.join(source_dir, "edit_prompt")

    if not all(os.path.isdir(d) for d in [input_dir, edited_dir, prompt_dir]):
        raise FileNotFoundError("Required subdirectories not found in source_dir.")

    # Get list of images
    try:
        image_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        num_images = len(image_files)
        logger.info(f"Found {num_images} potential input images.")
        if num_images == 0:
            raise ValueError("No image files found in input_image directory.")
    except Exception as e:
        logger.error(f"Error listing files in {input_dir}: {e}")
        raise

    # Calculate or use provided map_size
    if map_size_gb is None:
        map_size = calculate_dataset_size(source_dir)
        logger.info(f"Automatically calculated map size: {map_size / (1024**3):.2f} GB")
    else:
        map_size = int(map_size_gb * 1024**3)
        logger.info(f"Using provided map size: {map_size_gb} GB")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    # Create LMDB environment with larger map size
    env = None
    try:
        env = lmdb.open(
            lmdb_path,
            map_size=map_size,
            metasync=False,  # Faster writes
            sync=True,       # But ensure data is written
            map_async=True   # Faster writes
        )
        
        written_count = 0
        skipped_count = 0

        with env.begin(write=True) as txn:
            for idx, img_file in enumerate(tqdm(image_files, desc="Converting to LMDB")):
                base_name = os.path.splitext(img_file)[0]
                key = base_name

                input_img_path = os.path.join(input_dir, img_file)
                edited_img_path = os.path.join(edited_dir, img_file)
                prompt_path = os.path.join(prompt_dir, f"{base_name}.txt")

                # Skip if files don't exist
                if not all(os.path.exists(p) for p in [input_img_path, edited_img_path, prompt_path]):
                    skipped_count += 1
                    continue

                try:
                    # Read prompt
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        edit_prompt = f.read().strip()
                    
                    # Validate prompt format
                    if not re.match(r"^\s*f\s*\d+,?\s*", edit_prompt, re.IGNORECASE):
                        skipped_count += 1
                        continue

                    # Read and verify images
                    for img_path in [input_img_path, edited_img_path]:
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                            # Quick image verification
                            PILImage.open(io.BytesIO(img_bytes)).verify()

                    # Store data
                    data_tuple = (
                        open(input_img_path, 'rb').read(),
                        open(edited_img_path, 'rb').read(),
                        edit_prompt
                    )
                    serialized_data = pickle.dumps(data_tuple)
                    txn.put(key.encode('utf-8'), serialized_data)
                    written_count += 1

                except Exception as e:
                    logger.warning(f"Error processing item {key}: {str(e)}")
                    skipped_count += 1
                    continue

    except lmdb.Error as e:
        logger.error(f"LMDB Error: {e}")
        raise
    finally:
        if env:
            env.close()

    # Report results
    logger.info("LMDB Creation Summary:")
    logger.info(f"- Items successfully written: {written_count}")
    logger.info(f"- Items skipped: {skipped_count}")
    
    if os.path.exists(os.path.join(lmdb_path, 'data.mdb')):
        final_size = os.path.getsize(os.path.join(lmdb_path, 'data.mdb'))
        logger.info(f"- Final LMDB size: {final_size / (1024**2):.2f} MB")

# --- How to run it ---
if __name__ == "__main__":
    # --- Configuration ---
    SOURCE_DATA_DIR = 'data/DoodlePixV6' # e.g., /content/drive/MyDrive/datasets/my_image_data
    OUTPUT_LMDB_PATH = 'data/DoodlePixV6_lmdb' # e.g., /content/drive/MyDrive/datasets/my_image_data.lmdb OR /content/my_image_data.lmdb if copying locally first

    create_lmdb_dataset(SOURCE_DATA_DIR, OUTPUT_LMDB_PATH)