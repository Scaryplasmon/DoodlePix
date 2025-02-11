import PIL
import torch
import gc
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline
import logging
from contextlib import nullcontext
import numpy as np
import os
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_pipeline(model_id):
    """Setup pipeline with memory optimizations"""
    try:
        # Enable memory efficient attention
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            # Enable memory efficient attention
            use_memory_efficient_attention=True
        ).to("cuda")

        # Enable model CPU offloading
        pipe.enable_model_cpu_offload()
        
        # Enable attention slicing
        pipe.enable_attention_slicing(slice_size="auto")
        
        # Enable VAE slicing
        pipe.enable_vae_slicing()
        
        # Optional: Enable xformers for more memory efficiency
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()

        return pipe
    except Exception as e:
        logger.error(f"Failed to setup pipeline: {str(e)}")
        raise

def generate_image(pipe, prompt, input_image, **kwargs):
    """Generate image with error handling and automatic memory management"""
    try:
        # Use autocast for mixed precision inference
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Set reasonable defaults for memory-constrained systems
            random_seed = random.randint(0, 1000000)
            params = {
                "num_inference_steps": 20,  # Lower number of steps
                "image_guidance_scale": 1.5,
                "guidance_scale": 7.5,
                "generator": torch.Generator("cuda").manual_seed(random_seed)
            }
            params.update(kwargs)
            
            result = pipe(
                prompt,
                image=input_image,
                **params
            )
            
            # Force CUDA synchronization
            torch.cuda.synchronize()
            
            return result.images[0]
            
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory! Try reducing batch size or image size")
        torch.cuda.empty_cache()
        gc.collect()
        raise
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Comprehensive image preprocessing to ensure compatibility
    """
    try:
        # Open and convert image to RGB mode
        if isinstance(image_path, str):
            image = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError("Input must be a file path or PIL Image")

        # Convert to RGB if image is in different mode (e.g., RGBA, L)
        if image.mode != 'RGB':
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')

        # Resize image while maintaining aspect ratio
        if target_size:
            image.thumbnail(target_size, PIL.Image.Resampling.LANCZOS)
            
            # Create new image with padding to exact target size
            new_image = Image.new('RGB', target_size, (255, 255, 255))  # white background
            paste_pos = ((target_size[0] - image.size[0]) // 2,
                        (target_size[1] - image.size[1]) // 2)
            new_image.paste(image, paste_pos)
            image = new_image

        # Convert to numpy array and normalize
        image_array = np.array(image)
        
        # Check and fix potential issues
        if image_array.max() > 255:
            logger.warning("Image values exceed 255, normalizing...")
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
            
        # Convert back to PIL
        image = Image.fromarray(image_array)

        # Verify final image properties
        logger.info(f"Processed image size: {image.size}, mode: {image.mode}")
        
        return image

    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise
import random
def main():
    try:
        model_id = "DoodlePix_V1/"
        image_path = "doodle5.png"
        output_path = "edited_image6.png"
        
        # Enhanced preprocessing
        input_image = preprocess_image(
            image_path,
            target_size=(512, 512)  # Adjust size based on your needs
        )
        
        # Optional: Save preprocessed image for debugging
        input_image.save("preprocessed_" + image_path)
        
        prompt = "<subject: strong Man>, <style: fantasy, Hercules UI icon>, <colors:foreground: pink and white, background: dark blue>, <details:very strong man with a beard and big muscles>"

        # Setup pipeline
        pipe = setup_pipeline(model_id)
        random_seed = random.randint(0, 1000000)
        
        # Generate image with timeout
        edited_image = generate_image(
            pipe,
            prompt,
            input_image,
            num_inference_steps=20,
            image_guidance_scale=0.5,
            guidance_scale=7.5,
            seed=random_seed
        )
        
        # Save result
        edited_image.save(output_path)
        logger.info(f"Successfully saved image to {output_path}")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()