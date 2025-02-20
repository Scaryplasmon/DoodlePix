from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Custom dataset for batch processing
class ImageFolder(Dataset):
    def __init__(self, folder_path, transform):
        self.files = list(Path(folder_path).glob('*.png'))  # Adjust extension if needed
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            image_path = self.files[idx]
            image = Image.open(image_path).convert('RGB')  # Force RGB mode
            return self.transform(image), str(image_path)
        except Exception as e:
            print(f"Error loading image {self.files[idx]}: {str(e)}")
            # Return a black image as fallback
            dummy = Image.new('RGB', (512, 512), 'black')
            return self.transform(dummy), str(image_path)

def determine_background_color(mask_array):
    # Check points near corners (10 pixels from corners)
    h, w = mask_array.shape
    points = [
        (10, 10),
        (10, w-10),
        (h-10, 10),
        (h-10, w-10)
    ]
    values = [mask_array[y, x] for y, x in points]
    mean_value = np.mean(values)
    return 'white' if mean_value > 0.5 else 'black'

def process_batch(batch_results, output_folder):
    """Process and save a batch of results"""
    for result in batch_results:
        try:
            output_image, output_path = result
            output_image.save(output_path)
        except Exception as e:
            print(f"Error saving image {output_path}: {str(e)}")

# Move all the execution code into main()
def main():
    # Setup model
    model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    torch.set_float32_matmul_precision('high')
    model.to('cuda')
    model.eval()

    # Data settings
    image_size = (512, 512)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Setup input/output folders
    input_folder = "train2/edited_image"
    output_folder = "bg_removed"
    os.makedirs(output_folder, exist_ok=True)

    # Create dataset and dataloader
    dataset = ImageFolder(input_folder, transform_image)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True)

    # Batch processing variables
    batch_results = []
    batch_count = 0
    save_frequency = 100  # Save every 100 images

    # Process images
    with torch.no_grad():
        try:
            for batch_images, batch_paths in tqdm(dataloader, desc="Processing images"):
                batch_images = batch_images.to('cuda')
                preds = model(batch_images)[-1].sigmoid().cpu()
                
                for pred, img_path in zip(preds, batch_paths):
                    try:
                        # Load original image
                        original_image = Image.open(img_path).convert('RGB')
                        
                        # Process mask
                        pred = pred.squeeze()
                        mask_pil = transforms.ToPILImage()(pred)
                        mask = mask_pil.resize(original_image.size)
                        
                        # Determine background color
                        # mask_array = np.array(mask)
                        bg_color = "white"
                        
                        # Create new image with alpha channel
                        result = original_image.copy()
                        result.putalpha(mask)
                        
                        # Create background
                        bg = Image.new('RGB', original_image.size, bg_color)
                        
                        # Composite
                        final_image = Image.alpha_composite(bg.convert('RGBA'), result.convert('RGBA')).convert('RGB')
                        
                        # Save path
                        filename = os.path.basename(img_path)
                        new_filename = f"{bg_color}_{filename}"
                        output_path = os.path.join(output_folder, new_filename)
                        
                        batch_results.append((final_image, output_path))
                        batch_count += 1

                        # Save batch if we've reached the frequency
                        if batch_count >= save_frequency:
                            process_batch(batch_results, output_folder)
                            batch_results = []
                            batch_count = 0
                            
                    except Exception as e:
                        print(f"Error processing image {img_path}: {str(e)}")
                        continue

            # Process any remaining images
            if batch_results:
                process_batch(batch_results, output_folder)
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted. Saving remaining images...")
            if batch_results:
                process_batch(batch_results, output_folder)
            print("Saved remaining images. Exiting...")

if __name__ == '__main__':
    main()