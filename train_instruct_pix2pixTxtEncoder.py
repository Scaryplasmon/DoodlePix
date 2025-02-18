#!/usr/bin/env python
"""
Script to fine-tune a CLIP text encoder with multi-task heads.
This version uses the edited image (colored version) for target extraction,
applies differential learning rates, and performs an improved validation.
It also includes a helper to validate on an actual prompt text file.

Example prompt:
  f8, [3D], <tags: head, human hair, girl, human nose, human body>, #ce9062, #79492d, #fbd891, #1d1c20, #ffffff background.

This script uses your original load_image_dataset function (which loads the images and prompts)
and pads the palette to a fixed maximum length (--max_palette_length).

Usage example:
  accelerate launch train_instruct_pix2pixTxtEncoder.py --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" --train_data_dir="DoodlePixV5_WIP/" --max_train_steps=10000 --lambda_bg=0.5 --batch_size 4 --gradient_accumulation_steps 8
"""

import argparse
import os
import re
import math
import logging
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import Dataset as HFDataset, Image as DsImage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Original dataset loading function
# ---------------------------
def load_image_dataset(source_dir):
    input_dir = os.path.join(source_dir, "input_image")
    edited_dir = os.path.join(source_dir, "edited_image")
    prompt_dir = os.path.join(source_dir, "edit_prompt")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dataset_dict = {"input_image": [], "edited_image": [], "edit_prompt": []}
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        input_img_path = os.path.join(input_dir, img_file)
        edited_img_path = os.path.join(edited_dir, img_file)
        prompt_path = os.path.join(prompt_dir, f"{base_name}.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            edit_prompt = f.read().strip()
        dataset_dict["input_image"].append(input_img_path)
        dataset_dict["edited_image"].append(edited_img_path)
        dataset_dict["edit_prompt"].append(edit_prompt)
    dataset = HFDataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("input_image", DsImage())
    dataset = dataset.cast_column("edited_image", DsImage())
    return dataset

# ---------------------------
# Helper functions for parsing and color conversion
# ---------------------------
def hex_to_rgb(hex_str):
    """Convert hex color string (e.g. '#ce9062') to normalized RGB tensor."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        raise ValueError("Invalid hex color")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return torch.tensor([r, g, b], dtype=torch.float32)

def extract_background_color(image, border_width=10):
    """Estimate the background color by averaging the border pixels."""
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    border_pixels = []
    for x in range(width):
        for y in [0, height - 1]:
            border_pixels.append(pixels[x, y])
    for y in range(1, height - 1):
        for x in [0, width - 1]:
            border_pixels.append(pixels[x, y])
    avg = [sum([p[i] for p in border_pixels]) / len(border_pixels) for i in range(3)]
    return torch.tensor(avg, dtype=torch.float32) / 255.0

def parse_prompt(prompt):
    """
    Parse the prompt into its components.
      - Fidelity: e.g. "f8" -> 8.0
      - Shading: from bracketed expression, e.g. "[3D]" -> "3d"
      - Tags: from <tags: ...> (list of strings)
      - Hex codes: all occurrences of hex color codes
    """
    fidelity_match = re.search(r'^\s*f(\d+)', prompt, re.IGNORECASE)
    fidelity = float(fidelity_match.group(1)) if fidelity_match else 0.0

    shading_match = re.search(r'\[([^\]]+)\]', prompt)
    shading_str = shading_match.group(1).strip().lower() if shading_match else "normal"
    shading_mapping = {"flat": 0, "3d": 1, "normal": 2, "outline": 3}
    shading = shading_mapping.get(shading_str, 2)

    tags_match = re.search(r'<tags:\s*([^>]+)>', prompt, re.IGNORECASE)
    tags = [tag.strip() for tag in tags_match.group(1).split(",")] if tags_match else []

    hex_codes = re.findall(r'#[0-9A-Fa-f]{6}', prompt)

    return fidelity, shading, tags, hex_codes

def process_palette(hex_codes, max_palette_length):
    """
    Convert a list of hex codes (of variable length) into a tensor of shape (max_palette_length, 3)
    and a boolean mask indicating valid positions.
    If fewer than max_palette_length codes are present, pad with zeros.
    If more, truncate to max_palette_length.
    """
    colors = [hex_to_rgb(code) for code in hex_codes]
    num = len(colors)
    if num < max_palette_length:
        for _ in range(max_palette_length - num):
            colors.append(torch.zeros(3, dtype=torch.float32))
    elif num > max_palette_length:
        colors = colors[:max_palette_length]
        num = max_palette_length
    palette_tensor = torch.stack(colors)  # (max_palette_length, 3)
    mask = torch.zeros(max_palette_length, dtype=torch.bool)
    mask[:num] = True
    return palette_tensor, mask

# ---------------------------
# Dataset
# ---------------------------
class TextMultiTaskDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_palette_length=10, resolution=512):
        """
        data_dir: path containing subfolders "input_image" and "edit_prompt"
        tokenizer: CLIPTokenizer
        max_palette_length: maximum number of colors to predict (padded if necessary)
        resolution: image resolution for background extraction
        """
        self.dataset = load_image_dataset(data_dir)
        self.tokenizer = tokenizer
        self.max_palette_length = max_palette_length
        self.resolution = resolution

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Use the edited image (colored image) for target extraction
        edited_image = sample["edited_image"]
        prompt = sample["edit_prompt"]

        # Resize the edited image to the expected resolution
        edited_image = edited_image.resize((self.resolution, self.resolution))

        fidelity, shading, tags, hex_codes = parse_prompt(prompt)
        target_palette, palette_mask = process_palette(hex_codes, self.max_palette_length)
        # Extract background color from the edited image
        target_bg = extract_background_color(edited_image)

        tokenized = self.tokenizer(
            prompt, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_fidelity": torch.tensor([fidelity], dtype=torch.float32),
            "target_shading": torch.tensor(shading, dtype=torch.long),
            "target_palette": target_palette,
            "palette_mask": palette_mask,
            "target_bg": target_bg
        }

# ---------------------------
# Model definition: multi-task head
# ---------------------------
class TextEncoderMultiTask(nn.Module):
    def __init__(self, text_encoder: CLIPTextModel, max_palette_length: int, num_shading_classes: int = 4):
        """
        Wraps a CLIP text encoder and adds:
          - a fidelity head (regression)
          - a shading head (classification)
          - a palette head (predicts a fixed-length sequence of colors)
        """
        super().__init__()
        self.text_encoder = text_encoder  # full precision
        hidden_size = text_encoder.config.hidden_size
        self.fidelity_head = nn.Linear(hidden_size, 1)
        self.shading_head = nn.Linear(hidden_size, num_shading_classes)
        self.palette_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_palette_length * 3)
        )
        self.max_palette_length = max_palette_length

    def forward(self, input_ids, attention_mask=None):
        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        fidelity_pred = self.fidelity_head(cls_emb)
        shading_logits = self.shading_head(cls_emb)
        palette_pred = self.palette_head(cls_emb)
        palette_pred = palette_pred.view(-1, self.max_palette_length, 3)
        return fidelity_pred, shading_logits, palette_pred

# ---------------------------
# Helper function for validation between the starting and finetuned encoder
# ---------------------------
def validate_text_encoder(original_encoder, finetuned_encoder, tokenizer, sample_prompts, device):
    """
    For each prompt, compute the cosine similarity between 
    the [CLS] representation of the original and finetuned text encoder.
    """
    from torch.nn.functional import cosine_similarity
    validation_results = {}
    original_encoder.eval()
    finetuned_encoder.eval()
    with torch.no_grad():
        for prompt in sample_prompts:
            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            orig_out = original_encoder(**tokenized).last_hidden_state[:, 0, :]
            fine_out = finetuned_encoder(**tokenized).last_hidden_state[:, 0, :]
            cos_sim = cosine_similarity(orig_out, fine_out).item()
            l2_dist = torch.norm(orig_out - fine_out, p=2).item()
            validation_results[prompt] = {"cosine_similarity": cos_sim, "l2_distance": l2_dist}
    original_encoder.train()
    finetuned_encoder.train()
    return validation_results

def validate_sample_prompt(filepath, model, tokenizer, device):
    """
    Validate model predictions on a specific prompt from a file.
    This function reads the prompt from the file, tokenizes it,
    and passes it through the multi-task model to output predicted fidelity, shading, and palette.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    model.eval()
    with torch.no_grad():
        fidelity_pred, shading_logits, palette_pred = model(tokenized["input_ids"], tokenized["attention_mask"])
        predictions = {
            "fidelity": fidelity_pred.squeeze().item(),
            "shading": torch.argmax(shading_logits, dim=-1).squeeze().item(),
            "palette": palette_pred.cpu().tolist(),
        }
    model.train()
    return prompt, predictions

def validate_multitask_output(model, tokenizer, sample_prompts, device):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for prompt in sample_prompts:
            tokenized = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            fidelity_pred, shading_logits, palette_pred = model(tokenized["input_ids"], tokenized["attention_mask"])
            predictions[prompt] = {
                "fidelity": fidelity_pred.squeeze().item(),
                "shading": torch.argmax(shading_logits, dim=-1).squeeze().item(),
                "palette": palette_pred.cpu().tolist(),
            }
    model.train()
    return predictions

def validate_from_prompt_file(filepath, original_encoder, finetuned_encoder, tokenizer, device):
    with open(filepath, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    original_encoder.eval()
    finetuned_encoder.eval()
    with torch.no_grad():
        orig_embedding = original_encoder(**tokenized).last_hidden_state[:, 0, :].cpu().numpy()
        fine_embedding = finetuned_encoder(**tokenized).last_hidden_state[:, 0, :].cpu().numpy()
    original_encoder.train()
    finetuned_encoder.train()
    return prompt, orig_embedding, fine_embedding

# ---------------------------
# Main training function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path or identifier of the pretrained CLIP text encoder")
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Directory with 'input_image', 'edited_image', and 'edit_prompt' subfolders")
    parser.add_argument("--max_palette_length", type=int, default=6,
                        help="Maximum number of hex colors (palette length)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--lambda_bg", type=float, default=0.5,
                        help="Weight for background color loss")
    parser.add_argument("--output_dir", type=str, default="models/txtEncoder/")
    parser.add_argument("--sample_prompt_file", type=str, default="DoodlePixV5_WIP/edit_prompt/3D_00002_bis.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of gradient accumulation steps")
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="no")  # using FP32 in this example
    device = accelerator.device

    # Load tokenizer and text encoder in FP32
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    base_text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    base_text_encoder.to(device=device)
    
    # Save a copy of the starting (original) text encoder for validation comparison later.
    original_text_encoder = copy.deepcopy(base_text_encoder)  # used only for comparisons

    # Build multi-task model around the text encoder
    model = TextEncoderMultiTask(base_text_encoder, max_palette_length=args.max_palette_length)
    model.to(device=device)

    # Create the dataset and DataLoader (assuming your TextMultiTaskDataset is defined)
    dataset = TextMultiTaskDataset(args.train_data_dir, tokenizer, max_palette_length=args.max_palette_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Set up optimizer and loss functions
    optimizer = optim.AdamW([
        {"params": model.text_encoder.parameters(), "lr": args.learning_rate},
        {"params": model.fidelity_head.parameters()},
        {"params": model.shading_head.parameters()},
        {"params": model.palette_head.parameters()}
    ], lr=args.learning_rate)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    model.train()
    global_step = 0
    accumulation_steps = args.gradient_accumulation_steps
    accumulation_counter = 0
    sample_prompts = [
        "f8, [3d], <tags: sample prompt 1>",
        "f8, [3d], <tags: sample prompt 2>"
    ]

    while global_step < args.max_train_steps:
        for batch in dataloader:
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_fidelity = batch["target_fidelity"].to(device)
            target_shading = batch["target_shading"].to(device)
            target_palette = batch["target_palette"].to(device)
            palette_mask = batch["palette_mask"].to(device)
            target_bg = batch["target_bg"].to(device)

            fidelity_pred, shading_logits, palette_pred = model(input_ids, attention_mask=attention_mask)
            
            # Compute the different losses
            loss_fidelity = mse_loss(fidelity_pred, target_fidelity)
            loss_shading = ce_loss(shading_logits, target_shading)
            # Process palette loss
            B, L, _ = palette_pred.shape
            pred_flat = palette_pred.view(B * L, 3)
            target_flat = target_palette.view(B * L, 3)
            mask_flat = palette_mask.view(B * L).unsqueeze(-1).float()
            valid_count = mask_flat.sum() + 1e-8
            loss_palette = (mse_loss(pred_flat * mask_flat, target_flat * mask_flat) * mask_flat.numel()) / valid_count
            # Background loss (averaging across the batch)
            loss_bg = 0.0
            for i in range(B):
                valid = palette_mask[i].nonzero(as_tuple=False)
                if valid.numel() > 0:
                    last_idx = valid[-1].item()
                    pred_bg = palette_pred[i, last_idx, :]
                    # Get the target background: squeeze if necessary to remove extra dimensions.
                    target_bg_val = target_bg[i] if B > 1 else target_bg
                    if target_bg_val.dim() > 1:  # Remove extra dimension if present
                        target_bg_val = target_bg_val.squeeze(0)
                    loss_bg += mse_loss(pred_bg, target_bg_val)
            loss_bg = loss_bg / B

            total_loss = loss_fidelity + loss_shading + loss_palette + args.lambda_bg * loss_bg

            # Normalize loss for accumulation
            total_loss = total_loss / accumulation_steps
            accelerator.backward(total_loss)
            accumulation_counter += 1

            # When we've accumulated enough gradients, update the model
            if accumulation_counter % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log a basic loss every 100 updates
                if global_step % 100 == 0:
                    logger.info(f"Step {global_step}: "
                                f"Fidelity Loss = {loss_fidelity.item():.4f}, "
                                f"Shading Loss = {loss_shading.item():.4f}, "
                                f"Palette Loss = {loss_palette.item():.4f}, "
                                f"BG Loss = {loss_bg.item():.4f}, "
                                f"Total Loss = {total_loss.item() * accumulation_steps:.4f}")

                # Every 5 or 1000 global steps, save metrics and validation results
                if global_step % 1000 == 0 or global_step == 500:
                    # Ensure the output directory exists
                    os.makedirs(args.output_dir, exist_ok=True)
                    
                    validation_metrics = validate_text_encoder(
                        original_text_encoder, model.text_encoder, tokenizer, sample_prompts, device
                    )
                    multitask_preds = validate_multitask_output(model, tokenizer, sample_prompts, device)
                    
                        # Run additional validation using an actual prompt text file (if provided)
                    if args.sample_prompt_file:
                        # Validate using the text encoder comparison first.
                        prompt_text, orig_emb, fine_emb = validate_from_prompt_file(
                            args.sample_prompt_file, original_text_encoder, model.text_encoder, tokenizer, device
                        )
                        logger.info("Validation on sample prompt file: " + args.sample_prompt_file)
                        logger.info("Prompt: " + prompt_text)
                        logger.info("Original Embedding (first 5 values): " + str(orig_emb[0][:5]))
                        logger.info("Fine-tuned Embedding (first 5 values): " + str(fine_emb[0][:5]))

                        # Now validate using the multi-task prediction from the full model.
                        prompt_text, sample_preds = validate_sample_prompt(args.sample_prompt_file, model, tokenizer, device)
                        logger.info("Multi-task predictions on sample prompt file:")
                        logger.info(json.dumps(sample_preds, indent=4))
                        
                    metrics = {
                        "global_step": global_step,
                        "loss_fidelity": loss_fidelity.item(),
                        "loss_shading": loss_shading.item(),
                        "loss_palette": loss_palette.item(),
                        "loss_bg": loss_bg.item(),
                        "total_loss": total_loss.item() * accumulation_steps,
                        "validation_metrics": validation_metrics,
                        "multitask_predictions": multitask_preds,
                        "sample_prompt": args.sample_prompt_file,
                        "sample_prompt_predictions": sample_preds
                    }
                    metrics_path = os.path.join(args.output_dir, f"metrics_{global_step}.json")
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=4)
                    logger.info(f"Saved improved metrics at step {global_step} to {metrics_path}")

                # Check termination condition
                if global_step >= args.max_train_steps:
                    break

    # Save the final fine-tuned multi-task text encoder
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        os.makedirs(args.output_dir, exist_ok=True)
        # Save only the underlying text_encoder in HF format
        unwrapped_model.text_encoder.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")

    # Run additional validation using an actual prompt text file (if provided)
    if args.sample_prompt_file:
        prompt_text, orig_emb, fine_emb = validate_from_prompt_file(args.sample_prompt_file, original_text_encoder, model.text_encoder, tokenizer, device)
        logger.info("Validation on sample prompt file:")
        logger.info(f"Prompt: {prompt_text}")
        logger.info(f"Original Embedding (first 5 values): {orig_emb[0][:5]}")
        logger.info(f"Fine-tuned Embedding (first 5 values): {fine_emb[0][:5]}")

if __name__ == "__main__":
    main()
