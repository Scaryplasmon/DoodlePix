"""
This script trains a style embedding via textual inversion for the InstructPix2Pix pipeline.
It adds a new token (e.g. "<my_style>") to the tokenizer and updates its embedding in the text encoder,
while freezing other parameters. The training objective is similar to textual inversion: given an input
image, we encode it with the VAE, add noise, and compute an MSE loss between the UNet noise prediction 
and the actual noise. The prompt passed to the text encoder contains the special style token.
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_scheduler

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionInstructPix2PixPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StyleDataset(Dataset):
    """
    A dataset that loads images from a folder, applies a simple augmentation,
    and resizes them to the specified resolution.
    """
    def __init__(self, image_dir, resolution):
        self.image_dir = image_dir
        self.resolution = resolution
        self.image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        # Simple augmentation: random horizontal flip
        if np.random.rand() > 0.5:
            image = ImageOps.mirror(image)
        image = image.resize((self.resolution, self.resolution))
        image = np.array(image).astype(np.float32) / 127.5 - 1.0  # scale to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # CHW format
        return image


def add_style_token(tokenizer, text_encoder, style_token, initializer_token):
    """
    Add a new special token to the tokenizer and initialize its embedding using the embedding of
    an initializer token.
    """
    if style_token in tokenizer.get_vocab():
        logger.info(f"Token {style_token} already exists in tokenizer.")
    else:
        logger.info(f"Adding token {style_token} to the tokenizer.")
        tokenizer.add_tokens([style_token])
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(style_token)
        initializer_id = tokenizer.convert_tokens_to_ids(initializer_token)
        with torch.no_grad():
            text_encoder.get_input_embeddings().weight[token_id].copy_(
                text_encoder.get_input_embeddings().weight[initializer_id]
            )
    return tokenizer.convert_tokens_to_ids(style_token)


def freeze_parameters(model, train_embedding=False):
    """
    Freeze all parameters in the model. If train_embedding is True, keep the embedding layer trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    if train_embedding:
        # Unfreeze only the input embedding weights (we assume our target token is part of these)
        for name, param in model.named_parameters():
            if "embeddings" in name or "input_embedding" in name:
                logger.info(f"Training parameter: {name}")
                param.requires_grad = True


def log_validation(vae, unet, text_encoder, tokenizer, scheduler, args, device, global_step, style_token_id):
    """Run validation with multiple style strengths."""
    logger.info("\nRunning validation...")
    
    # Use several style scales for testing
    style_scales = [1.0, 2.0, 3.0]
    
    validation_dir = os.path.join(args.train_data_dir, "val")
    if not os.path.exists(validation_dir):
        print(f"Validation directory not found: {validation_dir}")
        return
    
    # Create a new pipeline for validation
    validation_pipeline = StableDiffusionInstructPix2PixPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)
    
    for img_file in os.listdir(validation_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        base_name = os.path.splitext(img_file)[0]
        input_path = os.path.join(validation_dir, img_file)
        prompt_path = os.path.join(validation_dir, f"{base_name}.txt")
        
        init_image = Image.open(input_path).convert("RGB")
        init_image = init_image.resize((args.resolution, args.resolution))
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                original_prompt = f.read().strip()
        else:
            original_prompt = "a photo"
        
        for style_scale in style_scales:
            try:
                # Create a prompt that includes repeated style token(s)
                style_token_part = " ".join([args.style_token] * args.style_token_repeats)
                styled_prompt = f"{original_prompt}, in {style_token_part} style (strength: {style_scale})"
                print(f"\nValidating {base_name} with style scale {style_scale}")
                
                with torch.no_grad():
                    text_inputs = tokenizer(
                        styled_prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    text_embeddings = text_encoder(text_inputs.input_ids)[0]
                    
                    # Apply style conditioning: multiply only the positions where the style token appears.
                    style_mask = (text_inputs.input_ids == style_token_id).unsqueeze(-1)
                    text_embeddings = torch.where(style_mask, text_embeddings * style_scale, text_embeddings)
                
                result = validation_pipeline(
                    prompt=styled_prompt,
                    image=init_image,
                    num_inference_steps=20,
                    guidance_scale=7,
                    image_guidance_scale=1.5,
                    generator=torch.Generator(device=device).manual_seed(42),
                    text_embeddings=text_embeddings  # Pass precomputed embeddings
                )
                output_image = result.images[0]
                
                out_dir = os.path.join(args.output_dir, "validation_outputs", f"step_{global_step}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{base_name}_style{style_scale:.1f}.png")
                output_image.save(out_path)
                print(f"Saved validation output to: {out_path}")
            except Exception as e:
                print(f"Validation error on {base_name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Train style embedding via textual inversion for InstructPix2Pix.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path or model identifier of the pretrained model.")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Directory with training images.")
    parser.add_argument("--output_dir", type=str, default="style_embedding_model",
                        help="Where to save the style embedding.")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Resolution of input images.")
    parser.add_argument("--num_train_steps", type=int, default=1000,
                        help="Number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate for the style token embedding.")
    parser.add_argument("--style_token", type=str, default="<my_style>",
                        help="The new style token to learn.")
    parser.add_argument("--initializer_token", type=str, default="a",
                        help="A token used to initialize the style token embedding.")
    parser.add_argument("--validate_every", type=int, default=500,
                        help="Run validation every N training steps.")
    parser.add_argument("--style_conditioning_scale", type=float, default=5.0,
                        help="Scale factor for style token during training")
    parser.add_argument("--style_scale", type=float, default=2.5,
                        help="Scale factor for style token during inference")
    parser.add_argument("--style_token_repeats", type=int, default=5,
                        help="Number of times to repeat style token in prompt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.to(device)

    # Add the style token and obtain its id
    style_token_id = add_style_token(tokenizer, text_encoder, args.style_token, args.initializer_token)
    logger.info(f"Style token id: {style_token_id}")

    # Freeze parameters except the input embeddings (train only the new token)
    freeze_parameters(text_encoder, train_embedding=True)

    # Setup dataset and dataloader with augmentation
    dataset = StyleDataset(args.train_data_dir, args.resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load the VAE, UNet, and noise scheduler.
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Create a parameter for the style token embedding
    embedding_weights = text_encoder.get_input_embeddings()
    optimized_token = torch.nn.Parameter(embedding_weights.weight[style_token_id].clone())
    
    # Set up the optimizer on the optimized token only
    optimizer = optim.AdamW([optimized_token], lr=args.learning_rate, weight_decay=0.01)
    num_warmup_steps = 100
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.num_train_steps,
    )

    progress_bar = tqdm(total=args.num_train_steps, desc="Training", dynamic_ncols=True)
    global_step = 0
    text_encoder.train()

    loss_ema = 0
    beta = 0.95

    while global_step < args.num_train_steps:
        for image in dataloader:
            image = image.to(device)
            # Encode the image to get latents
            with torch.no_grad():
                encoder_out = vae.encode(image.float())
                latents = encoder_out.latent_dist.sample() * 0.18215

            # Update the style token embedding in the text encoder with our optimized token
            embedding_weights.weight.data[style_token_id] = optimized_token.data

            # Create the prompt with repeated style token
            style_tokens = " ".join([args.style_token] * args.style_token_repeats)
            prompt = f"a photo in {style_tokens} style"

            # Tokenize and get text embeddings
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
            style_mask = (text_inputs.input_ids == style_token_id).unsqueeze(-1)
            text_embeddings = torch.where(
                style_mask,
                text_embeddings * args.style_conditioning_scale,
                text_embeddings
            )

            # Sample noise and create noisy latents
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            model_input = torch.cat([latents, noisy_latents], dim=1)
            noise_pred = unet(model_input, timesteps, encoder_hidden_states=text_embeddings).sample
            loss = F.mse_loss(noise_pred, noise)

            loss_ema = beta * loss_ema + (1 - beta) * loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([optimized_token], max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()

            # Log the current loss and norm of the optimized token
            token_norm = optimized_token.norm().item()
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "EMA Loss": f"{loss_ema:.4f}",
                "Token Norm": f"{token_norm:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            progress_bar.update(1)
            global_step += 1

            if global_step % args.validate_every == 0:
                log_validation(vae, unet, text_encoder, tokenizer, scheduler, args, device, global_step, style_token_id)

            if global_step >= args.num_train_steps:
                break

    progress_bar.close()

    # Save the updated text encoder and tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_encoder.save_pretrained(output_dir / "text_encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    logger.info("Training finished. Saved style embedding model.")


if __name__ == "__main__":
    main()
