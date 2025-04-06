import os
import lmdb
import shutil
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_lmdb(lmdb_path, target_size_mb, buffer_increment_mb=64):
    """
    Create a resized copy of an LMDB database with a target size.
    Original database is never modified.
    
    Args:
        lmdb_path (str): Path to the original LMDB database
        target_size_mb (int): Target size in megabytes
        buffer_increment_mb (int): Size increment in MB for each retry if needed
    
    Returns:
        tuple: (success (bool), final_size_mb (int), new_path (str))
    """
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB database not found at: {lmdb_path}")

    # Get original size
    data_path = os.path.join(lmdb_path, 'data.mdb')
    original_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    logger.info(f"Original LMDB size: {original_size_mb:.2f} MB")

    # Create new path for resized database
    new_path = f"{lmdb_path}_resized_{target_size_mb}mb"
    
    current_attempt_mb = target_size_mb
    max_attempts = 10
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        current_size_bytes = int(current_attempt_mb * 1024 * 1024)
        
        logger.info(f"Attempt {attempt}: Trying size {current_attempt_mb:.2f}MB")
        
        # Initialize environments as None
        src_env = None
        dst_env = None
        
        try:
            # Clean up any existing database
            if os.path.exists(new_path):
                logger.info(f"Removing existing resized database at {new_path}")
                shutil.rmtree(new_path)
                time.sleep(0.5)  # Give OS time to release handles
            
            # Open source environment
            src_env = lmdb.open(lmdb_path, readonly=True, max_readers=1)
            
            # Create new environment with target size
            dst_env = lmdb.open(
                new_path,
                map_size=current_size_bytes,
                metasync=False,
                sync=True,
                map_async=True
            )

            # Copy data
            with src_env.begin(write=False) as src_txn:
                with dst_env.begin(write=True) as dst_txn:
                    cursor = src_txn.cursor()
                    for key, value in cursor:
                        dst_txn.put(key, value)

            # Close environments properly
            if src_env:
                src_env.close()
                src_env = None
            if dst_env:
                dst_env.close()
                dst_env = None

            final_size_mb = os.path.getsize(os.path.join(new_path, 'data.mdb')) / (1024 * 1024)
            logger.info(f"Successfully created resized LMDB at {new_path} ({final_size_mb:.2f}MB)")
            return True, final_size_mb, new_path

        except lmdb.MapFullError:
            logger.warning(f"Map full error at {current_attempt_mb}MB, increasing size...")
            # Close environments before cleanup
            if src_env:
                src_env.close()
            if dst_env:
                dst_env.close()
            time.sleep(0.5)  # Give OS time to release handles
            
            # Clean up failed attempt
            if os.path.exists(new_path):
                try:
                    shutil.rmtree(new_path)
                except PermissionError:
                    logger.warning("Could not remove temporary database immediately, will try on next iteration")
            
            current_attempt_mb += buffer_increment_mb
            continue

        except Exception as e:
            logger.error(f"Error during resize: {str(e)}")
            # Close environments before cleanup
            if src_env:
                src_env.close()
            if dst_env:
                dst_env.close()
            time.sleep(0.5)  # Give OS time to release handles
            
            # Clean up failed attempt
            if os.path.exists(new_path):
                try:
                    shutil.rmtree(new_path)
                except PermissionError:
                    logger.warning("Could not remove temporary database due to file handles still being open")
            return False, original_size_mb, None

        finally:
            # Ensure environments are closed in all cases
            if src_env:
                src_env.close()
            if dst_env:
                dst_env.close()

    logger.error(f"Failed to resize after {max_attempts} attempts")
    return False, original_size_mb, None

if __name__ == "__main__":
    # Example usage
    LMDB_PATH = "data/DoodlePixV6_lmdb"
    TARGET_SIZE_MB = 1240  # Desired size
    BUFFER_INCREMENT_MB = 6  # Increment if needed

    success, final_size, new_path = resize_lmdb(LMDB_PATH, TARGET_SIZE_MB, BUFFER_INCREMENT_MB)
    
    if success:
        logger.info(f"Resized database created at: {new_path}")
        logger.info(f"Final size: {final_size:.2f}MB")
    else:
        logger.error("Resize operation failed")