"""Script to train an IP-Adapter for style control in InstructPix2Pix pipeline."""

import argparse
import inspect
import logging
import math
import os
from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_scheduler
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_image_dataset(source_dir):
    input_dir = os.path.join(source_dir, "input_image")
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    dataset_dict = {
        "input_image": [],
        "edited_image": [],
        "edit_prompt": []
    }
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        input_img_path = os.path.join(source_dir, "input_image", img_file)
        edited_img_path = os.path.join(source_dir, "edited_image", img_file)
        prompt_path = os.path.join(source_dir, "edit_prompt", f"{base_name}.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            edit_prompt = f.read().strip()
        dataset_dict["input_image"].append(input_img_path)
        dataset_dict["edited_image"].append(edited_img_path)
        dataset_dict["edit_prompt"].append(edit_prompt)
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("input_image", Image())
    dataset = dataset.cast_column("edited_image", Image())
    return dataset


class IPAdapter(nn.Module):
    """IP-Adapter module with stronger architecture"""
    def __init__(self, cross_attention_dim=768, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.clip_extra_context_tokens = clip_extra_context_tokens
        hidden_dim = 1024  # Increased hidden dimension
        self.image_projection = nn.Sequential(
            nn.Linear(clip_embeddings_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cross_attention_dim)
        )
        self.style_scale = nn.Parameter(torch.ones(1, 1, cross_attention_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, clip_extra_context_tokens, cross_attention_dim) * 0.02)

    def forward(self, clip_embeddings):
        projected = self.image_projection(clip_embeddings)  # [batch, cross_attention_dim]
        projected = projected.unsqueeze(1).repeat(1, self.clip_extra_context_tokens, 1)
        projected = projected + self.pos_embedding
        projected = projected * self.style_scale
        return projected


class StyleDataset(Dataset):
    """Dataset for IP-Adapter style training using InstructPix2Pix data format"""
    def __init__(self, data_dir, resolution=512):
        self.resolution = resolution
        self.data_dir = data_dir
        self.input_dir = os.path.join(data_dir, "input_image")
        self.image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.model_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.clip_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        base_name = os.path.splitext(image_name)[0]
        input_path = os.path.join(self.data_dir, "input_image", image_name)
        input_image = Image.open(input_path).convert('RGB')
        input_image = self.model_transform(input_image)
        edited_path = os.path.join(self.data_dir, "edited_image", image_name)
        edited_image = Image.open(edited_path).convert('RGB')
        style_image_model = self.model_transform(edited_image)
        style_image_clip = self.clip_transform(edited_image)
        prompt_path = os.path.join(self.data_dir, "edit_prompt", f"{base_name}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        else:
            prompt = ""
        return {"input_image": input_image, "style_image": style_image_model,
                "style_image_clip": style_image_clip, "prompt": prompt}


#############################################
# Patching the Pipeline
#############################################

def patch_unet_forward(unet):
    """Patch UNet.forward to inject ip_adapter_conditioning by concatenating it to the text conditioning."""
    original_forward = unet.forward

    def new_forward(sample, timestep, encoder_hidden_states, *args, **kwargs):
        ip_cond = kwargs.pop("ip_adapter_conditioning", None)
        if ip_cond is None and hasattr(unet, "parent_pipe") and getattr(unet.parent_pipe, "_ip_adapter_conditioning", None) is not None:
            ip_cond = unet.parent_pipe._ip_adapter_conditioning
        if ip_cond is not None:
            ip_cond = ip_cond.to(encoder_hidden_states.dtype)
            # Instead of replacing, concatenate along the sequence dimension.
            encoder_hidden_states = torch.cat([encoder_hidden_states, ip_cond], dim=1)
        return original_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)

    unet._original_forward = original_forward
    unet.forward = new_forward
    return unet


def patch_pipeline_call(pipe):
    """Patch pipeline __call__ to accept ip_adapter_conditioning and store it on the pipeline."""
    original_call = pipe.__call__

    def new_call(*args, ip_adapter_conditioning=None, **kwargs):
        pipe._ip_adapter_conditioning = ip_adapter_conditioning
        result = original_call(*args, **kwargs)
        pipe._ip_adapter_conditioning = None
        return result

    pipe.__call__ = new_call.__get__(pipe, type(pipe))
    return pipe

def patch_pipe(pipe, ip_adapter):
    """Patch the pipeline to use IP-Adapter with weighted combination"""
    
    def new_unet_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        ip_conditioning = kwargs.pop("ip_adapter_conditioning", None)
        if ip_conditioning is not None:
            dtype = encoder_hidden_states.dtype
            ip_conditioning = ip_conditioning.to(dtype=dtype)
            
            # Use weighted combination instead of replacement
            alpha = 0.5  # Can be made configurable
            encoder_hidden_states = encoder_hidden_states * (1 - alpha) + ip_conditioning * alpha
        
        return self._original_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)
    
    pipe.unet._original_forward = pipe.unet.forward
    pipe.unet.forward = new_unet_forward.__get__(pipe.unet)
    return pipe


#############################################
# Training and Validation

#############################################

def save_training_logs(log_data, output_dir):
    """Save training logs to JSON file"""
    log_file = os.path.join(output_dir, "training_logs.json")
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

def log_step_metrics(ip_adapter, noise_loss, content_loss_val, global_step):
    """Get metrics for current training step"""
    metrics = {
        "step": global_step,
        "noise_loss": noise_loss.item(),
        "content_loss": content_loss_val.item(),
        "total_loss": (noise_loss + content_loss_val).item(),
    }
    
    # Log gradient norms for key components
    if hasattr(ip_adapter, "style_scale") and ip_adapter.style_scale.grad is not None:
        metrics["style_scale_grad_norm"] = ip_adapter.style_scale.grad.norm().item()
    
    # Log last layer gradients
    last_layer = list(ip_adapter.image_projection.children())[-1]
    if last_layer.weight.grad is not None:
        metrics["last_layer_grad_norm"] = last_layer.weight.grad.norm().item()
    
    # Log IP adapter output norms
    if hasattr(ip_adapter, "_last_output"):
        metrics["ip_adapter_output_norm"] = ip_adapter._last_output.norm().item()
    
    return metrics

def train_ip_adapter():
    parser = argparse.ArgumentParser(description="Train IP-Adapter for style control")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--validate_every", type=int, default=500,
                        help="Number of steps between validation runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    ).to(device)

    image_encoder = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=weight_dtype
    ).to(device)

    ip_adapter = IPAdapter(
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4
    ).to(device).to(dtype=weight_dtype)

    # Patch the pipeline so that during validation the adapter output is concatenated to the text conditioning.
    pipe = patch_pipe(pipe, ip_adapter)

    dataset = StyleDataset(args.train_data_dir, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    content_loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(ip_adapter.parameters(), lr=args.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(dataloader))
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=max_train_steps // 10,
        num_training_steps=max_train_steps,
    )

    progress_bar = tqdm(range(max_train_steps), desc="Training IP-Adapter")
    global_step = 0
    training_logs = []

    for epoch in range(args.num_train_epochs):
        ip_adapter.train()
        epoch_loss = 0
        
        for batch in dataloader:
            input_images = batch["input_image"].to(device, dtype=weight_dtype)
            style_images = batch["style_image"].to(device, dtype=weight_dtype)
            style_images_clip = batch["style_image_clip"].to(device, dtype=weight_dtype)
            
            with torch.no_grad():
                style_embeddings = image_encoder(style_images_clip).last_hidden_state[:, 0]
            
            # Get IP conditioning
            ip_conditioning = ip_adapter(style_embeddings)
            
            # Print norms for debugging
            if global_step % 100 == 0:
                print(f"\nStep {global_step} - IP conditioning norm: {ip_conditioning.norm().item():.4f}")
            
            # Compute latents and noise
            latents = pipe.vae.encode(input_images).latent_dist.sample() * 0.18215
            target_latents = pipe.vae.encode(style_images).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Forward pass
            model_input = torch.cat([latents, noisy_latents], dim=1)
            noise_pred = pipe.unet(
                model_input,
                timesteps,
                encoder_hidden_states=ip_conditioning,
            ).sample
            
            # Compute losses
            noise_loss = F.mse_loss(noise_pred.float(), noise.float())
            content_loss_val = content_loss(latents, target_latents)
            loss = noise_loss + 2.0 * content_loss_val
            
            # Backward pass
            loss.backward()
            
            # Log metrics
            metrics = log_step_metrics(ip_adapter, noise_loss, content_loss_val, global_step)
            training_logs.append(metrics)
            
            # Save logs periodically
            if global_step % 100 == 0:
                save_training_logs(training_logs, args.output_dir)
                print(f"\nSaved training logs at step {global_step}")
                print("Latest metrics:")
                for k, v in metrics.items():
                    print(f"{k}: {v:.6f}")
            
            # Optimization step
            torch.nn.utils.clip_grad_norm_(ip_adapter.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Epoch": epoch})
            progress_bar.update(1)
            
            # Run validation periodically
            if global_step % args.validate_every == 0:
                log_validation(pipe, ip_adapter, image_encoder, args, device, global_step)
            
            global_step += 1
    
    # Save final logs
    save_training_logs(training_logs, args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ip_adapter.state_dict(), output_dir / "ip_adapter.pt")
    logger.info("Training finished. Saved IP-Adapter weights and logs.")


def log_validation(pipe, ip_adapter, image_encoder, args, device, global_step):
    """Run validation with IP-Adapter using a reference style"""
    logger.info("\nRunning validation...")
    
    # Get reference style image
    style_image_path = os.path.join(args.train_data_dir, "edited_image", os.listdir(os.path.join(args.train_data_dir, "edited_image"))[0])
    style_image = Image.open(style_image_path).convert("RGB")
    
    # Transform for CLIP
    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                           (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # Get style embeddings
    style_image_clip = clip_transform(style_image).unsqueeze(0).to(device)
    with torch.no_grad():
        style_embeddings = image_encoder(style_image_clip).last_hidden_state[:, 0]
    
    # Try different style scales
    style_scales = [0.0, 0.5, 1.0, 2.0]
    
    # Validation directory
    # the next time u change the path of a directory ure gonna die of illness u stupid fuckking bitch
    validation_dir = os.path.join(args.train_data_dir, "val")
    if os.path.exists(validation_dir):
        for img_file in os.listdir(validation_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            base_name = os.path.splitext(img_file)[0]
            input_path = os.path.join(validation_dir, img_file)
            prompt_path = os.path.join(validation_dir, f"{base_name}.txt")
            
            # Load input image
            init_image = Image.open(input_path).convert("RGB")
            init_image = init_image.resize((args.resolution, args.resolution))
            
            # Get prompt
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            else:
                prompt = "a photo"
            
            for style_scale in style_scales:
                try:
                    print(f"\nTesting with style scale {style_scale} on {base_name}")
                    print(f"Using prompt: {prompt}")
                    
                    with torch.no_grad():
                        ip_conditioning = ip_adapter(style_embeddings)
                    
                    result = pipe(
                        prompt=prompt,
                        image=init_image,
                        num_inference_steps=20,
                        guidance_scale=7.0,
                        image_guidance_scale=1.5,
                        ip_adapter_scale=style_scale,
                        ip_adapter_conditioning=ip_conditioning,
                        generator=torch.Generator(device=device).manual_seed(42),
                    ).images[0]
                    
                    # Save output
                    output_dir = os.path.join(args.output_dir, "validation_outputs", f"step_{global_step}")
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"{base_name}_style{style_scale:.1f}_output.png")
                    result.save(output_path)
                    print(f"Saved output to: {output_path}")
                except Exception as e:
                    print(f"Error during inference on {base_name}: {str(e)}")
                    print(f"Full error: {e.__class__.__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
    else:
        print(f"Validation directory not found: {validation_dir}")


if __name__ == "__main__":
    train_ip_adapter()
