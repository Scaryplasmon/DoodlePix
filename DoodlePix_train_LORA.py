#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Script to fine-tune Stable Diffusion for LORA InstructPix2Pix.
    Base code referred from: https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py
"""

import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import json
import PIL.Image
import PIL.ImageOps
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
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from DoodlePix_pipeline import StableDiffusionInstructPix2PixPipeline

from fidelity_mlp import FidelityMLP
from pytorch_msssim import ssim
import lpips
import re
if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
    fidelity_mlp=None,
):
    try:
        logger.info("Running validation...")
        
        validation_dir = os.path.join(args.output_dir, "validation")
        current_step = accelerator.step
        step_dir = os.path.join(validation_dir, f"step_{current_step}")
        os.makedirs(step_dir, exist_ok=True)
        
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        val_images_dir = args.val_images_dir
        try:
            image_files = [f for f in os.listdir(val_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        except (FileNotFoundError, OSError) as e:
            logger.error(f"Error accessing validation directory: {e}")
            return
        
        logger.info(f"Found {len(image_files)} validation images")
        
        edited_images_dict = {}
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        fidelity_levels = [2, 8]
        
        with autocast_ctx:
            for image_file in image_files:
                try:
                    base_name = os.path.splitext(image_file)[0]
                    image_path = os.path.join(val_images_dir, image_file)
                    prompt_path = os.path.join(val_images_dir, f"{base_name}.txt")
                    
                    if not os.path.exists(prompt_path):
                        logger.warning(f"Skipping {image_file} - no corresponding prompt file found")
                        continue
                        
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        validation_prompt = f.read().strip()
                        
                    original_image = PIL.Image.open(image_path).convert("RGB")
                    
                    try:
                        original_image.save(os.path.join(step_dir, f"original_{base_name}.png"))
                    except Exception as e:
                        logger.error(f"Failed to save original image {base_name}: {e}")
                        continue

                    edited_images = []
                    for fidelity in fidelity_levels:
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        fidelity_prompt = f"f{fidelity} {validation_prompt}"
                        
                        try:
                            edited_image = pipeline(
                                fidelity_prompt,
                                negative_prompt="NSFW, nudity, bad, blurry, sex",
                                image=original_image,
                                num_inference_steps=24,
                                image_guidance_scale=1.5,
                                guidance_scale=6.0,
                                generator=generator,
                                safety_checker=None
                            ).images[0]
                            
                            edited_image.save(os.path.join(step_dir, f"edited_{base_name}_f{fidelity}.png"))
                            edited_images.append(edited_image)
                            
                            with open(os.path.join(step_dir, f"prompt_{base_name}_f{fidelity}.txt"), "w") as f:
                                f.write(fidelity_prompt)
                                
                        except Exception as e:
                            logger.error(f"Failed to generate/save edited image {base_name} with fidelity {fidelity}: {e}")
                            continue
                    
                    if edited_images:
                        edited_images_dict[base_name] = {
                            "original": original_image,
                            "edited": edited_images,
                            "prompt": validation_prompt
                        }
                        
                except Exception as e:
                    logger.error(f"Failed to process validation image {image_file}: {e}")
                    continue

        try:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES + ["Fidelity"])
                    for base_name, data in edited_images_dict.items():
                        for i, edited_image in enumerate(data["edited"]):
                            fidelity = fidelity_levels[i % len(fidelity_levels)]
                            wandb_table.add_data(
                                wandb.Image(data["original"]),
                                wandb.Image(edited_image),
                                data["prompt"],
                                f"f{fidelity}"
                            )
                    tracker.log({"validation": wandb_table})
        except Exception as e:
            logger.error(f"Failed to log to wandb: {e}")
            
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.info("Continuing training despite validation failure")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--no_proxy",
        default=False,
        action="store_true",
        help=(
            "Whether to use proxy loss for fidelity or actual fidelity loss"
        ),
    )
    parser.add_argument(
        "--fidelity_weight",
        type=float,
        default=0.1,
        help="Weight for the fidelity loss.",
    )
    parser.add_argument(
        "--fidelity_loss_type",
        type=str,
        default="l1",
        choices=["l1", "ssim", "lpips"],
        help="Type of loss for the full fidelity calculation ('l1', 'ssim', 'lpips'). Only used if --no_proxy is True.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--val_images_dir",
        type=str,
        default=None,
        help="Directory containing validation images and their corresponding .txt prompt files",
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
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps. Set to -1 to disable during training and only run at the end.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_fidelity_control", 
        action="store_true",
        help="Whether to use fidelity control through FidelityMLP",
    )
    parser.add_argument(
        "--style_strength",
        type=float,
        default=0.7,
        help="Strength of style transfer (0.0-1.0) - higher values preserve less of original content",
    )
    parser.add_argument(
        "--save_optimizer_state",
        action="store_true",
        help="Whether to save optimizer state with checkpoints for resuming training",
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



def compute_fidelity_loss(
    pred_images: torch.Tensor,
    input_images: torch.Tensor,
    target_images: torch.Tensor,
    fidelity_values: torch.Tensor,
    loss_type: str = "l1",
    lpips_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Compute a fidelity-weighted loss balancing input preservation and edit strength.

    Args:
        pred_images: Predicted images from the model [B, C, H, W], range [-1, 1] or [0, 1].
        input_images: Original input images [B, C, H, W], range [-1, 1] or [0, 1].
        target_images: Target edited images [B, C, H, W], range [-1, 1] or [0, 1].
        fidelity_values: Batch of fidelity control values (normalized to [0, 1]). [B]
        loss_type: Type of loss ('l1', 'ssim', 'lpips').
        lpips_model: Initialized LPIPS model (required if loss_type is 'lpips').
        device: The torch device to run calculations on.

    Returns:
        A loss tensor.
    """
    pred_images = pred_images.to(device=device, dtype=torch.float32)
    input_images = input_images.to(device=device, dtype=torch.float32)
    target_images = target_images.to(device=device, dtype=torch.float32)
    fidelity_values = fidelity_values.to(device=device, dtype=torch.float32).view(-1) # Ensure shape [B]

    batch_size = pred_images.shape[0]
    dist_to_input = torch.zeros(batch_size, device=device)
    dist_to_target = torch.zeros(batch_size, device=device)

    if loss_type == "l1":
        dist_to_input = torch.abs(pred_images - input_images).mean(dim=[1, 2, 3])
        dist_to_target = torch.abs(pred_images - target_images).mean(dim=[1, 2, 3])

    elif loss_type == "ssim":

        pred_images_01 = (pred_images + 1) / 2 if pred_images.min() < 0 else pred_images
        input_images_01 = (input_images + 1) / 2 if input_images.min() < 0 else input_images
        target_images_01 = (target_images + 1) / 2 if target_images.min() < 0 else target_images

        for i in range(batch_size):
            dist_to_input[i] = 1.0 - ssim(pred_images_01[i].unsqueeze(0), input_images_01[i].unsqueeze(0), data_range=1.0, size_average=True)
            dist_to_target[i] = 1.0 - ssim(pred_images_01[i].unsqueeze(0), target_images_01[i].unsqueeze(0), data_range=1.0, size_average=True)

    elif loss_type == "lpips":
        if lpips_model is None:
            raise ValueError("lpips_model must be provided for loss_type 'lpips'")
        print("the device is", device)
        pred_images_norm = pred_images if pred_images.min() >= -1 else (pred_images * 2) - 1
        input_images_norm = input_images if input_images.min() >= -1 else (input_images * 2) - 1
        target_images_norm = target_images if target_images.min() >= -1 else (target_images * 2) - 1

        dist_to_input = lpips_model(pred_images_norm, input_images_norm).squeeze()
        dist_to_target = lpips_model(pred_images_norm, target_images_norm).squeeze()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    weighted_loss = (fidelity_values * dist_to_input) + ((1 - fidelity_values) * dist_to_target)

    return weighted_loss.mean()




def load_image_dataset(source_dir):
    """
    Load an image dataset from a directory structured like:
    source_dir/
        input_image/
            image1.png
            image2.png
            ...
        edited_image/
            image1.png
            image2.png
            ...
        edit_prompt/
            image1.txt
            image2.txt
            ...
    """
    input_image_dir = os.path.join(source_dir, "input_image")
    edited_image_dir = os.path.join(source_dir, "edited_image")
    edit_prompt_dir = os.path.join(source_dir, "edit_prompt")
    
    if not os.path.exists(input_image_dir):
        raise ValueError(f"Input image directory {input_image_dir} does not exist")
    if not os.path.exists(edited_image_dir):
        raise ValueError(f"Edited image directory {edited_image_dir} does not exist")
    if not os.path.exists(edit_prompt_dir):
        raise ValueError(f"Edit prompt directory {edit_prompt_dir} does not exist")
    
    image_files = [f for f in os.listdir(input_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    data = {
        "input_image": [],
        "edited_image": [],
        "edit_prompt": []
    }
    
    for img_file in image_files:
        edited_img_path = os.path.join(edited_image_dir, img_file)
        
        base_name = os.path.splitext(img_file)[0]
        prompt_file = f"{base_name}.txt"
        prompt_path = os.path.join(edit_prompt_dir, prompt_file)
        
        if not os.path.exists(edited_img_path):
            logger.warning(f"Missing edited version for {img_file}, skipping")
            continue
        
        if not os.path.exists(prompt_path):
            logger.warning(f"Missing prompt file for {img_file}, skipping")
            continue
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except Exception as e:
            logger.warning(f"Error reading prompt file {prompt_path}: {e}, skipping")
            continue
        
        data["input_image"].append(PIL.Image.open(os.path.join(input_image_dir, img_file)).convert("RGB"))
        data["edited_image"].append(PIL.Image.open(edited_img_path).convert("RGB"))
        data["edit_prompt"].append(prompt)
    
    if len(data["input_image"]) == 0:
        raise ValueError(f"No valid image-prompt pairs found in {source_dir}")
        
    logger.info(f"Loaded {len(data['input_image'])} image-prompt pairs from {source_dir}")
    
    return datasets.Dataset.from_dict(data)

lpips_loss_fn = None
def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    if unet.conv_in.in_channels == 8:
        logger.info("UNet already has 8 input channels, skipping channel expansion")
    else:
        logger.info(f"Expanding UNet input channels from {unet.conv_in.in_channels} to 8")
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :unet.conv_in.in_channels, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # referred to https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    unet.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "conv_in", "conv_out"],
    )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        
    fidelity_mlp = None
    if args.use_fidelity_control:
        potential_mlp_path = os.path.join(args.pretrained_model_name_or_path, "fidelity_mlp")
        
        if os.path.exists(potential_mlp_path):
            try:
                logger.info(f"Loading existing FidelityMLP from {potential_mlp_path}")
                fidelity_mlp = FidelityMLP.from_pretrained(potential_mlp_path)
            except Exception as e:
                logger.warning(f"Failed to load FidelityMLP from {potential_mlp_path}: {e}. Creating new one.")
                hidden_size = text_encoder.config.hidden_size
                fidelity_mlp = FidelityMLP(hidden_size=hidden_size)
        else:
            logger.info("Creating new FidelityMLP.")
            hidden_size = text_encoder.config.hidden_size
            fidelity_mlp = FidelityMLP(hidden_size=hidden_size)
            
        fidelity_mlp.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    params_to_optimize = [
        {"params": unet_lora_parameters, "lr": args.learning_rate},
    ]

    if args.use_fidelity_control and fidelity_mlp is not None:
        mlp_lr = args.fidelity_mlp_learning_rate if hasattr(args, 'fidelity_mlp_learning_rate') and args.fidelity_mlp_learning_rate else args.learning_rate
        params_to_optimize.append({"params": fidelity_mlp.parameters(), "lr": mlp_lr})
        logger.info(f"Added FidelityMLP parameters to optimizer with LR: {mlp_lr}")
        trainable_params = unet_lora_parameters + list(fidelity_mlp.parameters())
    else:
        trainable_params = unet_lora_parameters

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()

                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    elif args.train_data_dir is not None:
        try:
            dataset = {"train": load_image_dataset(args.train_data_dir)}
        except Exception as e:
            logger.warning(f"Failed to load dataset using custom loader: {e}")
            data_files = {"train": os.path.join(args.train_data_dir, "**")}
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
    else:
        raise ValueError("Need either a dataset name or a training folder.")


    column_names = dataset["train"].column_names

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )


    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )

        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(examples):
        preprocessed_images = preprocess_images(examples)

        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions(captions)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
#-------ACCELERATOR PREPARE-------
    if args.use_fidelity_control and fidelity_mlp is not None:
            unet, fidelity_mlp, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, fidelity_mlp, optimizer, train_dataloader, lr_scheduler
            )
            logger.info("Prepared UNet, FidelityMLP, Optimizer, DataLoader, Scheduler with Accelerator.")
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        logger.info("Prepared UNet, Optimizer, DataLoader, Scheduler with Accelerator (No FidelityMLP).")

    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        if args.report_to == "none":
            accelerator.init_trackers("instruct-pix2pix", config=None)
        else:
            config_dict = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
            accelerator.init_trackers("instruct-pix2pix", config=config_dict)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if args.use_fidelity_control:
        fidelity_mlp = FidelityMLP(hidden_size=unet.config.cross_attention_dim)
        fidelity_mlp.to(accelerator.device)
        fidelity_mlp = accelerator.prepare(fidelity_mlp)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):

                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()


                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]


                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                
                fidelity_tensor_for_loss = None
                if args.use_fidelity_control and fidelity_mlp is not None:
                    fidelity_values = []
                    prompts_text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    for prompt_text in prompts_text:
                        match = re.search(r"f\s*=?\s*(\d+)|f(\d+)", prompt_text, re.IGNORECASE)
                        if match:
                            f_int = int(match.group(1) if match.group(1) else match.group(2))
                            f_int = max(0, min(f_int, 9))
                            fidelity_val = f_int / 9.0
                        else:
                            fidelity_val = 0.7
                        fidelity_values.append(fidelity_val)

                    fidelity_tensor = torch.tensor(fidelity_values, device=latents.device, dtype=latents.dtype).unsqueeze(-1)
                    
                    fidelity_embedding = fidelity_mlp(fidelity_tensor)

                    injection_scale = args.fidelity_weight
                    encoder_hidden_states[:, 0:1, :] = encoder_hidden_states[:, 0:1, :] + injection_scale * fidelity_embedding.unsqueeze(1)
                    
                    fidelity_tensor_for_loss = fidelity_tensor.squeeze(-1)

                    if accelerator.is_main_process and global_step % 10 == 0:
                        logger.info(f"Step {global_step}: Fidelity embedding stats: mean={fidelity_embedding.mean().item():.4f}, std={fidelity_embedding.std().item():.4f}")

                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    original_image_embeds = image_mask * original_image_embeds

                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                                    
                
                
# --- ADD FIDELITY LOSS ---
                if args.use_fidelity_control and fidelity_mlp is not None and fidelity_tensor_for_loss is not None:
                    fidelity_loss_component = torch.tensor(0.0, device=loss.device)

                    if args.no_proxy:
                        if global_step % 10 == 0:
                            with torch.no_grad():
                                pred_latents = noise_scheduler.step(model_pred.detach(), timesteps[0], noisy_latents, return_dict=False)[0]
                                pred_latents = pred_latents.to(dtype=weight_dtype)
                                pred_images = vae.decode(pred_latents / vae.config.scaling_factor, return_dict=False)[0]
                                input_images = batch["original_pixel_values"].to(device=pred_images.device, dtype=pred_images.dtype)
                                target_images = batch["edited_pixel_values"].to(device=pred_images.device, dtype=pred_images.dtype)

                                global lpips_loss_fn 
                                if args.fidelity_loss_type == 'lpips' and lpips_loss_fn is None:
                                        logger.info("Initializing LPIPS model for LoRA script...")
                                        lpips_loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)
                                        lpips_loss_fn.eval()

                                fidelity_loss_component = compute_fidelity_loss(
                                    pred_images=pred_images,
                                    input_images=input_images,
                                    target_images=target_images,
                                    fidelity_values=fidelity_tensor_for_loss,
                                    loss_type=args.fidelity_loss_type,
                                    lpips_model=lpips_loss_fn if args.fidelity_loss_type == 'lpips' else None,
                                    device=accelerator.device
                                )
                            if accelerator.is_main_process and global_step % 10 == 0:
                                logger.info(f"Step {global_step}: Using FULL fidelity loss ({args.fidelity_loss_type}): {fidelity_loss_component.item():.4f}")

                    else:
                        fidelity_scale = fidelity_tensor_for_loss.view(-1, 1, 1, 1)
                        fidelity_loss_component = torch.mean(torch.abs(model_pred) * fidelity_scale)
                        if accelerator.is_main_process and global_step % 100 == 0:
                           logger.info(f"Step {global_step}: Using PROXY fidelity loss: {fidelity_loss_component.item():.4f}")

                    fidelity_weight = min(args.fidelity_weight, args.fidelity_weight * (global_step / 200)) # Gradual increase/ more like a warmup
                    loss = loss + fidelity_weight * fidelity_loss_component
                
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(trainable_params)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (
                    args.val_images_dir is not None and 
                    (args.validation_steps > 0 and global_step % args.validation_steps == 0)
                ):
                    logger.info(f"Running validation at step {global_step}")
                    if args.use_ema:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    
                    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(unet),
                        text_encoder=unwrap_model(text_encoder),
                        vae=unwrap_model(vae),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    
                    accelerator.step = global_step
                    
                    log_validation(
                        pipeline,
                        args,
                        accelerator,
                        generator,
                        fidelity_mlp=fidelity_mlp if args.use_fidelity_control else None,
                    )

                    if args.use_ema:
                        ema_unet.restore(unet.parameters())

                    del pipeline
                    torch.cuda.empty_cache()

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionInstructPix2PixPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionInstructPix2PixPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.load_lora_weights(args.output_dir)

        images = None
        if (args.val_image_url is not None) and (args.validation_prompt is not None):
            images = log_validation(
                pipeline,
                args,
                accelerator,
                generator,
                fidelity_mlp=fidelity_mlp if args.use_fidelity_control else None,
            )
        # When saving the trained model, also save the FidelityMLP if used
        if args.use_fidelity_control and fidelity_mlp is not None and accelerator.is_main_process:
            fidelity_mlp_path = os.path.join(args.output_dir, "fidelity_mlp")
            os.makedirs(fidelity_mlp_path, exist_ok=True)
            unwrapped_fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)
            unwrapped_fidelity_mlp.save_pretrained(fidelity_mlp_path)
            logger.info(f"Saved FidelityMLP LoRA weights to {fidelity_mlp_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
