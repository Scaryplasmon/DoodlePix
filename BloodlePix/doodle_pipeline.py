import torch
import numpy as np
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
import cv2
import controlnet_hinter

import json
from datetime import datetime

def save_run_settings(settings_dict, filepath):
    """Save run settings to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(settings_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Failed to save settings: {str(e)}")
        return False

def load_run_settings(filepath):
    """Load run settings from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load settings: {str(e)}")
        return None

CONTROLNET_MAPPING = {
    "canny": {
        "hinter": controlnet_hinter.hint_canny
    },
    "scribble": {
        "hinter": controlnet_hinter.hint_scribble
    },
    "hed": {
        "hinter": controlnet_hinter.hint_hed
    }
}

def setup_pipeline(model_path):
    """Initialize the pipeline with optimizations"""
    try:
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            use_memory_efficient_attention=True,
            safety_checker=None
        ).to("cuda")
        
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing(slice_size="auto")
        pipe.enable_vae_slicing()
        
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            
        return pipe
    except Exception as e:
        print(f"Failed to setup pipeline: {str(e)}")
        raise

def process_image(image, is_drawing_mode=True, control_type="canny"):
    """Process input image based on mode"""
    if is_drawing_mode:
        # Invert drawing (black on white to white on black)
        return ImageOps.invert(image)
    else:
        # Apply control net processing
        return CONTROLNET_MAPPING[control_type]["hinter"](image)

def build_prompt(subject="", theme="", colors="", details="", is_doodle=False, style_mode="default"):
    """Build the complete prompt from components"""
    prompt_parts = []
    
    if is_doodle:
        prompt_parts.append("<doodle>")
        
    if style_mode != "default":
        prompt_parts.append(f"<mode: {style_mode}>")
        
    if subject:
        prompt_parts.append(f"<subject: {subject}>")
    if theme:
        prompt_parts.append(f"<theme: {theme}>")
    if colors:
        prompt_parts.append(f"<colors: {colors}>")
    if details:
        prompt_parts.append(f"<details: {details}>")
        
    return ", ".join(prompt_parts)

def generate_image(pipe, image, prompt, negative_prompt="NSFW, bad, sex, blurred, jpg, photorealistic, flares, blur, flare, porn", num_inference_steps=24, 
                  guidance_scale=3.0, image_guidance_scale=1.5, generator=None):
    """Generate image using the pipeline"""
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator
        ).images[0]
    
    return output
