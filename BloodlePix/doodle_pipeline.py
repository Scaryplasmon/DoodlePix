import torch
import numpy as np
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTextModel
from peft import PeftModel
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

SCHEDULER_MAP = {
    "DDIM": DDIMScheduler,
    "DDPM": DDPMScheduler,
    "PNDM": PNDMScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler,
}

def setup_pipeline(model_path, scheduler_name="DDIM", text_encoder_lora_path=None):
    """Initialize the pipeline with optimizations and custom components"""
    try:
        # Initialize base pipeline
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            use_memory_efficient_attention=True,
            safety_checker=None
        )
        
        # Set scheduler if specified
        if scheduler_name in SCHEDULER_MAP:
            pipe.scheduler = SCHEDULER_MAP[scheduler_name].from_config(pipe.scheduler.config)
        
        # Load and apply LoRA weights to text encoder if specified
        if text_encoder_lora_path:
            print(f"Loading LoRA text encoder from {text_encoder_lora_path}")
            base_text_encoder = pipe.text_encoder
            pipe.text_encoder = PeftModel.from_pretrained(
                base_text_encoder,
                text_encoder_lora_path,
                is_trainable=False
            )
        
        # Move to GPU and enable optimizations
        pipe = pipe.to("cuda")
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

def build_prompt(subject="", colors="", world_style="fantasy", complexity=3, perspective="3d", description=""):
    """Build the complete prompt from components"""
    prompt_parts = []
    
    if subject:
        prompt_parts.append(f"<s: {subject.lower()}>")
    if colors:
        prompt_parts.append(f"<c: {colors.lower()}>")
    if world_style:
        prompt_parts.append(f"<w: {world_style.lower()}>")
    if complexity is not None:
        prompt_parts.append(f"<k: {str(complexity)}>")
    if perspective:
        prompt_parts.append(f"<p: {perspective.lower()}>")
    if description:
        prompt_parts.append(f"<d: {description.lower()}>")
        
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
