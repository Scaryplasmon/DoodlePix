"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

"""The training verts around trying to teach model to generate images based on B&W doodles or drawings. so the input_image is a black background and white line drawing, the edited_image is the end result colored icon, the edit prompt is the text prompt that describes the edited image for the model to follow."""

import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import itertools
import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from datasets import Dataset, Image as DatasetImage
from PIL import Image as PILImage
from peft import PeftModel, LoraConfig, get_peft_model
import glob

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def log_validation(unet, pipeline, args, accelerator, global_step):
    logger.info("Running validation...")
    
    if accelerator.is_main_process:
        # Save the current LoRA weights
        unwrapped_unet = accelerator.unwrap_model(unet)
        lora_state_dict = unwrapped_unet.state_dict()
        os.makedirs(f"{args.output_dir}/checkpoint-{global_step}", exist_ok=True)
        torch.save(lora_state_dict, f"{args.output_dir}/checkpoint-{global_step}/lora_weights.safetensors")

        # Create a fresh pipeline and load the saved LoRA weights
        validation_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(accelerator.device)
        
        try:
            # Get the trained UNet with LoRA and fuse the weights
            unwrapped_unet = accelerator.unwrap_model(unet)
            validation_pipeline.unet = unwrapped_unet
            validation_pipeline.unet.fuse_lora()
            
            # Run validation
            validation_dir = os.path.join(args.train_data_dir, "validation")
            if os.path.exists(validation_dir):
                for img_file in os.listdir(os.path.join(validation_dir, "input_image")):
                    if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    base_name = os.path.splitext(img_file)[0]
                    input_path = os.path.join(validation_dir, "input_image", img_file)
                    prompt_path = os.path.join(validation_dir, "edit_prompt", f"{base_name}.txt")
                    
                    # Load validation image and prompt
                    init_image = PILImage.open(input_path).convert("RGB")
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()

                    # Run inference
                    image = validation_pipeline(
                        prompt=prompt,
                        image=init_image,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7,
                        generator=torch.Generator(device=accelerator.device).manual_seed(42)
                    ).images[0]

                    # Save output
                    os.makedirs(os.path.join(args.output_dir, "validation_outputs", f"step_{global_step}"), exist_ok=True)
                    image.save(os.path.join(args.output_dir, "validation_outputs", f"step_{global_step}", f"{base_name}_output.png"))

        finally:
            # Unfuse LoRA weights after validation
            if hasattr(validation_pipeline.unet, "unfuse_lora"):
                validation_pipeline.unet.unfuse_lora()
            del validation_pipeline
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--val_images_dir",
        type=str,
        default="validation_images",
        help="Directory containing validation images and their corresponding prompt files",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default="train/input_image/image00162.png",
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default="<subject:rocket>, <style: 3D UI icon>, <colors:blue, silver, black background>, <theme:glass window metal frame>, <details:the icon is a rocket, set against a dark background>", help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help="Run validation every X steps. Set to -1 to disable during training and only run at the end.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", default=True, help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--text_encoder_lora_path",
        type=str,
        default=None,
        help="Path to the LoRA text encoder model",
    )
    parser.add_argument(
        "--text_encoder_learning_rate",
        type=float,
        default=1e-5,  # Usually 5-10x smaller than unet learning rate
        help="Learning rate for text encoder fine-tuning"
    )
    parser.add_argument(
        "--text_encoder_teacher_loss_weight",
        type=float,
        default=0.1,
        help="Weight for teacher distillation loss"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation (lower=faster training, higher=more capacity)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=8,
        help="Alpha parameter for LoRA scaling",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        type=str,
        default=["q_proj", "k_proj", "v_proj", "out_proj", "to_k", "to_q", "to_v", "to_out.0"],
        help="Which modules to apply LoRA to",
    )
    parser.add_argument(
        "--style_prefix",
        type=str,
        default="in the style of",
        help="Prefix to add before style tokens in prompts",
    )
    parser.add_argument(
        "--style_name", 
        type=str,
        required=True,
        help="Name/description of the style being trained"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def load_image_dataset(source_dir):
    # Get all image files
    input_dir = os.path.join(source_dir, "input_image")
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    dataset_dict = {
        "input_image": [],
        "edited_image": [],
        "edit_prompt": []
    }
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        
        # Get full paths
        input_img_path = os.path.join(source_dir, "input_image", img_file)
        edited_img_path = os.path.join(source_dir, "edited_image", img_file)
        prompt_path = os.path.join(source_dir, "edit_prompt", f"{base_name}.txt")
        
        # Read prompt
        with open(prompt_path, 'r', encoding='utf-8') as f:
            edit_prompt = f.read().strip()
        
        dataset_dict["input_image"].append(input_img_path)
        dataset_dict["edited_image"].append(edited_img_path)
        dataset_dict["edit_prompt"].append(edit_prompt)
    
    # Create dataset and cast image columns
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("input_image", DatasetImage())
    dataset = dataset.cast_column("edited_image", DatasetImage())
    
    return dataset


def main():
    args = parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Load base pipeline
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
    )
    
    # Extract components and freeze
    vae = pipeline.vae.eval()
    text_encoder = pipeline.text_encoder.eval()
    unet = pipeline.unet
    tokenizer = pipeline.tokenizer
    noise_scheduler = pipeline.scheduler

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Load dataset
    dataset = load_image_dataset(args.train_data_dir)
    
    # Create transforms
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        # The images are already PIL images from the dataset
        images = [train_transforms(example) for example in examples["input_image"]]
        edited_images = [train_transforms(example) for example in examples["edited_image"]]
        
        # Tokenize prompts
        input_ids = tokenizer(
            examples["edit_prompt"],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return {
            "pixel_values": torch.stack(images),
            "edited_pixel_values": torch.stack(edited_images),
            "input_ids": input_ids,
        }

    # Create train dataset
    train_dataset = dataset.with_transform(preprocess_train)

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Configure LoRA for UNet - target the key layers for style transfer
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "conv1", "conv2", "conv_out",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to UNet only (text encoder stays frozen)
    unet = get_peft_model(unet, unet_lora_config)
    
    # Enable gradient checkpointing for memory efficiency
    unet.enable_gradient_checkpointing()

    # Move models to device and prepare for mixed precision
    vae = vae.to(accelerator.device, dtype=torch.float16)
    text_encoder = text_encoder.to(accelerator.device, dtype=torch.float16)
    unet = unet.to(accelerator.device)

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare for training
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            try:
                with accelerator.accumulate(unet):
                    # Clear GPU cache periodically
                    if step % 100 == 0:
                        torch.cuda.empty_cache()

                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        edited_latents = vae.encode(batch["edited_pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
                        edited_latents = edited_latents * vae.config.scaling_factor

                    # Add noise
                    noise = torch.randn_like(edited_latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                    noisy_latents = noise_scheduler.add_noise(edited_latents, noise, timesteps)

                    # Prepare input
                    model_input = torch.cat([latents, noisy_latents], dim=1)

                    # Get text embeddings
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict noise residual
                    model_pred = unet(model_input, timesteps, encoder_hidden_states).sample

                    # Calculate loss
                    target = noise
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Backpropagate
                    accelerator.backward(loss)
                    
                    # Clip gradients
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                # Log validation
                if global_step % args.validation_steps == 0:
                    log_validation(unet, pipeline, args, accelerator, global_step)

                if global_step >= args.max_train_steps:
                    break

            except Exception as e:
                logger.error(f"Error during training step: {e}")
                torch.cuda.empty_cache()
                continue

    # Save final LoRA weights
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        torch.save(
            unet.state_dict(),
            os.path.join(args.output_dir, "final_lora_weights.safetensors")
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
