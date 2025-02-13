"""
This script trains the UNet and the CLIP text encoder jointly for the Instruct Pix2Pix pipeline.
It uses differential learning rates and a weight anchoring regularization for the text encoder,
and saves checkpoints (including the text encoder) during training.
"""

import os
import re
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator, ProjectConfiguration
from tqdm.auto import tqdm
import logging
import shutil
from pathlib import Path

# Diffusers imports
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
)

logger = logging.getLogger(__name__)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Pretrained model name or path.")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Path to training data folder (should contain 'edited_pixel_values', 'original_pixel_values', 'edit_prompt').")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution.")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size (per device).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of steps to accumulate gradients.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing.")
    parser.add_argument("--max_train_steps", type=int, default=15000)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the UNet.")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Enable xformers memory efficient attention.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use 8-bit Adam optimizer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_flip", action="store_true",
                        help="Enable random flip during training.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and models.")
    parser.add_argument("--checkpointing_steps", type=int, default=5000,
                        help="Frequency (in steps) to save a checkpoint.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep.")

    # New arguments for text encoder fine-tuning
    parser.add_argument("--text_encoder_lr", type=float, default=1e-5,
                        help="Learning rate for text encoder fine-tuning.")
    parser.add_argument("--lambda_reg", type=float, default=0.001,
                        help="Regularization coefficient for anchoring text encoder weights.")

    args = parser.parse_args()
    return args


def load_dataset(args, tokenizer):
    """
    Loads your original dataset.
    The dataset is expected to yield a dictionary with:
       - "edited_pixel_values": preprocessed edited image tensor,
       - "original_pixel_values": preprocessed original image tensor,
       - "edit_prompt": the text prompt.
    Here we assume your existing loading method.
    """
    # For example, if you have an existing dataset loader function from diffusers:
    from diffusers.data import load_dataset  # replace if necessary with your own loader
    dataset = load_dataset(
        args.train_data_dir,
        size=args.resolution,
        tokenizer=tokenizer,
        random_flip=args.random_flip,
    )
    return dataset


def compute_text_encoder_reg_loss(current_model, original_state):
    """
    Computes an L2 loss between the current text encoder parameters and
    their original pre-trained values.
    """
    reg_loss = 0.0
    for name, param in current_model.named_parameters():
        if name in original_state:
            reg_loss += F.mse_loss(param, original_state[name].to(param.device))
    return reg_loss


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Starting training with args: %s", args)

    # Setup Accelerator with a project configuration so logs are saved under output_dir/logs
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load tokenizer and models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Optionally enable gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    # Freeze the VAE (it is not fine-tuned)
    vae.requires_grad_(False)

    # Save the original state of the text encoder (for weight anchoring)
    original_text_encoder_state = {k: v.clone().detach() for k, v in text_encoder.state_dict().items()}

    # Set up the noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load dataset & create DataLoader (using your original settings: the dataset yields "edit_prompt" etc.)
    train_dataset = load_dataset(args, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,  # change if needed
    )

    # Initialize optimizer with two parameter groups for UNet and text encoder
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        [
            {"params": unet.parameters(), "lr": args.learning_rate},
            {"params": text_encoder.parameters(), "lr": args.text_encoder_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Create a simple StepLR scheduler (adjust as needed)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    # Prepare models, optimizer, dataloader, and lr scheduler with accelerator
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)
    loss_list = []

    unet.train()
    text_encoder.train()

    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            # Use accelerator.accumulate to implement gradient accumulation
            with accelerator.accumulate(unet):
                # Process images: encode edited and original images using the VAE.
                edited_images = batch["edited_pixel_values"].to(accelerator.device)
                original_images = batch["original_pixel_values"].to(accelerator.device)
                weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

                latents = vae.encode(edited_images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Tokenize the edit prompt
                text_inputs = tokenizer(
                    batch["edit_prompt"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_inputs = {k: v.to(accelerator.device) for k, v in text_inputs.items()}
                text_encoder_output = text_encoder(**text_inputs, return_dict=True)
                text_embeddings = text_encoder_output.last_hidden_state

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get original image latents (for conditioning)
                orig_latents = vae.encode(original_images.to(dtype=weight_dtype)).latent_dist.sample()
                orig_latents = orig_latents * vae.config.scaling_factor

                # Concatenate the noisy latents with original image latents along the channel dimension
                concat_latents = torch.cat([noisy_latents, orig_latents], dim=1)

                # Forward pass through UNet with text conditioning
                model_pred = unet(concat_latents, timesteps, encoder_hidden_states=text_embeddings).sample

                # Compute main diffusion loss (MSE between the prediction and the noise)
                loss = F.mse_loss(model_pred.float(), noise.float())

                # Compute regularization loss to anchor text encoder weights
                reg_loss = compute_text_encoder_reg_loss(text_encoder, original_text_encoder_state)
                total_loss = loss + args.lambda_reg * reg_loss

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters()) + list(text_encoder.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            loss_list.append(total_loss.item())

            if global_step % 100 == 0 and accelerator.is_local_main_process:
                avg_loss = sum(loss_list[-100:]) / 100
                logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}")

            if global_step >= args.max_train_steps:
                break

        # Save checkpoint every checkpointing_steps in the main process.
        if accelerator.is_main_process() and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved checkpoint at {save_path}")

            # Save a pipeline that bundles UNet and the text encoder
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            pipeline.save_pretrained(args.output_dir)
            # Save text encoder separately as well
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            unwrapped_text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))

        if global_step >= args.max_train_steps:
            break

    # Final model save
    if accelerator.is_main_process():
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()