#!/usr/bin/env python
"""
Extract Palette Script

For each image in the input folder, this script:
  - Uses rembg to create an alpha mask.
  - Computes the background color as the average color of pixels where alpha < bg_alpha_thresh.
  - Extracts foreground pixels (alpha >= bg_alpha_thresh) and runs KMeans to get an initial palette.
  - Merges similar foreground colors in LAB space if they are too close.
  - Ensures the number of foreground colors is between min_colors and max_colors.
  - Saves a text file (with the same basename as the image) listing the hex codes for the foreground palette,
    and then the background color (with the word "background" appended).
  
Usage:
    python extract_palette.py --input_dir path/to/images --output_dir path/to/palettes
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from sklearn.cluster import KMeans
from tqdm import tqdm


def rgb_to_hex(rgb):
    """Convert an (R, G, B) tuple (0-255) to a hex string."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def rgb_to_lab(rgb):
    """
    Convert an RGB color (0-255) to LAB using OpenCV.
    Input: (R, G, B) tuple or array.
    Returns: LAB as a 3-element numpy array.
    """
    # OpenCV uses BGR order
    rgb_arr = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)[0][0]
    return lab.astype(np.float32)


def merge_similar_clusters(centers, counts, lab_threshold, min_colors):
    """
    Merge clusters that are closer than lab_threshold in LAB space.
    centers: list of RGB centers (as numpy arrays, 0-255)
    counts: corresponding counts (ints)
    lab_threshold: minimum distance in LAB space below which clusters are merged
    min_colors: do not merge below this number of clusters.
    
    Returns new lists: merged_centers, merged_counts.
    """
    centers = [np.array(c, dtype=np.float32) for c in centers]
    counts = list(counts)
    
    merged = True
    while merged and len(centers) > min_colors:
        merged = False
        n = len(centers)
        min_dist = None
        pair_to_merge = None
        # Compute pairwise distances (in LAB)
        for i in range(n):
            lab_i = rgb_to_lab(centers[i])
            for j in range(i+1, n):
                lab_j = rgb_to_lab(centers[j])
                dist = np.linalg.norm(lab_i - lab_j)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    pair_to_merge = (i, j)
        # If the closest pair is under the threshold, merge them
        if min_dist is not None and min_dist < lab_threshold:
            i, j = pair_to_merge
            # Weighted average of centers
            new_center = (centers[i] * counts[i] + centers[j] * counts[j]) / (counts[i] + counts[j])
            new_count = counts[i] + counts[j]
            # Remove the two clusters and add the merged one
            new_centers = []
            new_counts = []
            for idx in range(n):
                if idx not in pair_to_merge:
                    new_centers.append(centers[idx])
                    new_counts.append(counts[idx])
            new_centers.append(new_center)
            new_counts.append(new_count)
            # Update lists
            centers = new_centers
            counts = new_counts
            merged = True

    return centers, counts


def extract_palette(image_path, default_k, lab_threshold, min_colors, max_colors, bg_alpha_thresh=128, subsample=10000):
    """
    Extract a color palette from an image.
    - default_k: initial number of clusters for foreground.
    - lab_threshold: if the LAB distance between two clusters is below this, they will be merged.
    - min_colors: minimum number of foreground colors (after merging).
    - max_colors: maximum number of foreground colors.
    - bg_alpha_thresh: threshold on alpha (0-255) for foreground.
    - subsample: maximum number of foreground pixels to cluster.
    
    Returns a tuple (palette_colors, bg_hex) where palette_colors is a list of hex codes and bg_hex is the
    background hex code.
    """
    # Open image and ensure it's in RGB format
    try:
        orig_image = Image.open(image_path)
        if orig_image.mode != "RGB":
            orig_image = orig_image.convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return [], None

    orig_np = np.array(orig_image)  # (H, W, 3)
    H, W, _ = orig_np.shape
    
    # Remove background with rembg (get RGBA)
    try:
        result_image = remove(orig_image)
        result_image = result_image.convert("RGBA")
        result_np = np.array(result_image)  # (H, W, 4)
    except Exception as e:
        print(f"Error removing background from {image_path}: {e}")
        return [], None
    
    # Extract alpha channel
    alpha = result_np[:, :, 3]
    # Define masks: foreground = alpha >= bg_alpha_thresh, background = alpha < bg_alpha_thresh
    fg_mask = alpha >= bg_alpha_thresh
    bg_mask = ~fg_mask
    
    total_pixels = H * W
    # Compute background color (average of original image pixels where bg_mask is True)
    if np.sum(bg_mask) > 0:
        bg_pixels = orig_np[bg_mask]
        bg_color = np.mean(bg_pixels, axis=0)
        bg_color = np.clip(np.round(bg_color).astype(int), 0, 255)
    else:
        bg_color = None

    # Process foreground
    if np.sum(fg_mask) > 0:
        fg_pixels = orig_np[fg_mask]
        # Subsample if too many pixels
        if fg_pixels.shape[0] > subsample:
            idx = np.random.choice(fg_pixels.shape[0], subsample, replace=False)
            fg_pixels = fg_pixels[idx]
        # Run KMeans clustering with default_k clusters
        kmeans = KMeans(n_clusters=default_k, random_state=42).fit(fg_pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        counts = np.bincount(labels)
        # Sort clusters by counts descending
        sorted_idx = np.argsort(-counts)
        centers = [centers[i] for i in sorted_idx]
        counts = [counts[i] for i in sorted_idx]
        
        # Merge clusters that are too close (but do not merge below min_colors)
        merged_centers, merged_counts = merge_similar_clusters(centers, counts, lab_threshold, min_colors)
        
        # If after merging we have more than max_colors, attempt to merge further until we have max_colors.
        # Note: if no further merges are possible (clusters are noticeably distinct),
        # break out to avoid an infinite loop.
        while len(merged_centers) > max_colors:
            prev_count = len(merged_centers)
            merged_centers, merged_counts = merge_similar_clusters(merged_centers, merged_counts, lab_threshold, len(merged_centers)-1)
            if len(merged_centers) == prev_count:
                print("Warning: Unable to merge clusters further to reach max_colors for image", image_path)
                break
        
        # Now sort merged clusters by counts (most prevalent first)
        sorted_indices = np.argsort(-np.array(merged_counts))
        fg_palette = [merged_centers[i] for i in sorted_indices]
    else:
        fg_palette = []

    # Convert foreground palette centers to hex codes
    fg_hex = [rgb_to_hex(tuple(np.clip(np.round(c).astype(int), 0, 255))) for c in fg_palette]
    
    # Return the foreground palette and background color (if exists)
    bg_hex = rgb_to_hex(tuple(bg_color)) if bg_color is not None else None
    return fg_hex, bg_hex


def process_folder(input_dir, output_dir, default_k, lab_threshold, min_colors, max_colors, bg_alpha_thresh):
    """
    Process every image in the input folder and save its palette to a txt file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            fg_hex, bg_hex = extract_palette(str(img_path), default_k, lab_threshold, min_colors, max_colors, bg_alpha_thresh)
            out_file = output_dir / f"{img_path.stem}.txt"
            with open(out_file, "w") as f:
                for color in fg_hex:
                    f.write(color + "\n")
                if bg_hex is not None:
                    f.write(bg_hex + " background\n")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """comand example
        python palettePrompt.py -i DoodlePixV4/DoodlePixV5/A_outline/ -o DoodlePixV4/DoodlePixV5/A_outline/colors/
    """
    parser = argparse.ArgumentParser(description="Extract color palettes from images with dynamic merging.")
    parser.add_argument("--input_dir","-i", type=str, required=True, help="Path to the input folder with images.")
    parser.add_argument("--output_dir","-o", type=str, required=True, help="Path to the folder where palette txt files will be saved.")
    parser.add_argument("--default_k","-k", type=int, default=12, help="Initial number of clusters for KMeans on foreground pixels.")
    parser.add_argument("--lab_threshold","-t", type=float, default=60.0, help="Minimum LAB distance to consider two colors distinct.")
    parser.add_argument("--min_colors","-m", type=int, default=1, help="Minimum number of foreground colors in the palette.")
    parser.add_argument("--max_colors","-M", type=int, default=7, help="Maximum number of foreground colors in the palette.")
    parser.add_argument("--bg_alpha_thresh","-b", type=int, default=128, help="Alpha threshold (0-255) to define background.")
    args = parser.parse_args()
    
    process_folder(args.input_dir, args.output_dir, args.default_k, args.lab_threshold, args.min_colors, args.max_colors, args.bg_alpha_thresh)


if __name__ == "__main__":
    main()
