"""
    Custom implementation of InstructPix2Pix training script from the diffusers repo
    https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py

"""
import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
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
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from datasets import Dataset, Image
from PIL import Image as PILImage
import glob
from fidelity_mlpXL import FidelityMLP

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
    StableDiffusionXLInstructPix2PixPipeline,
)

if is_wandb_available():
    import wandb

check_min_version("0.33.0.dev0")

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
):
    try:
        logger.info("Running validation...")
        
        validation_dir = os.path.join(args.output_dir, "validation")
        current_step = accelerator.step
        step_dir = os.path.join(validation_dir, f"step_{current_step}")
        os.makedirs(step_dir, exist_ok=True)
        
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        # Get all validation images and their corresponding prompts
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

        fidelity_levels = [2, 8]  # Low, high fidelity
        
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
                        
                    original_image = PILImage.open(image_path).convert("RGB")
                    
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
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_txtEncoder_path",
        type=str,
        default=None,
        required=False,
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
        "--validation_steps",
        type=int,
        default=5000,
        help="Run validation every X steps. Set to -1 to disable during training and only run at the end.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", default=True, help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--fidelity_mlp_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the fidelity MLP",
    )
    parser.add_argument(
        "--fidelity_mlp_path",
        type=str,
        default=None,
        help="Path to fidelity MLP model or directory to save the model to.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training."
    )
    parser.add_argument(
        "--vae_precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help="VAE precision for mixed precision training stability."
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help="Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help="Coordinate for (the width) to be included in the crop coordinate embeddings needed by SDXL UNet."
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

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # SDXL requires two text encoders
    tokenizer_1 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    
    # Get the appropriate text encoder classes
    text_encoder_cls_1 = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    
    # Load the text encoders
    text_encoder_1 = text_encoder_cls_1.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_2 = text_encoder_cls_2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    
    # Group tokenizers and text_encoders
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]
    
    # Freeze text encoders
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # VAE loading with precision control
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    
    # Define torch dtype mapping for VAE precision
    TORCH_DTYPE_MAPPING = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    
    # Freeze VAE
    vae.requires_grad_(False)
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    
    # Load fidelity MLP if provided
    if args.fidelity_mlp_path:
        fidelity_mlp = FidelityMLP.from_pretrained(args.fidelity_mlp_path)
        fidelity_mlp.to(accelerator.device)
    else:
        fidelity_mlp = None
    
    # SDXL requires additional conditioning information
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def compute_time_ids(current_dtype=None):
        """Compute time IDs with the current weight dtype or specified dtype."""
        if current_dtype is None:
            current_dtype = weight_dtype
        
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        original_size = target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=current_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)
    
    def encode_prompt(text_encoders, tokenizers, prompt):
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)

            prompt_embeds = text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )

            # Handle different output structures between CLIP models
            if hasattr(prompt_embeds, "pooler_output"):
                pooled_prompt_embeds = prompt_embeds.pooler_output
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else: 
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
            
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
    def tokenize_captions(captions, tokenizer=None):
        """Tokenize captions with specified tokenizer or default to tokenizer_1."""
        if tokenizer is None:
            tokenizer = tokenizer_1
        
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    def encode_prompts(text_encoders, tokenizers, prompts):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []

        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)

        return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)
    
    def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(text_encoders, tokenizers, prompts)
            prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
            pooled_prompt_embeds_all = pooled_prompt_embeds_all.to(accelerator.device)
        return prompt_embeds_all, pooled_prompt_embeds_all
    
    # Compute null conditioning for classifier-free guidance
    def compute_null_conditioning():
        null_conditioning_list = []
        for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
            null_tokens = tokenize_captions([""], tokenizer=a_tokenizer)
            null_tokens = null_tokens.to(a_text_encoder.device)
            null_output = a_text_encoder(null_tokens, output_hidden_states=True)
            
            # Handle different output structures
            if hasattr(null_output, "hidden_states"):
                null_conditioning_list.append(null_output.hidden_states[-2])
            else:
                null_conditioning_list.append(null_output[2][-2])  # Access hidden states differently
            
        return torch.concat(null_conditioning_list, dim=-1)

    # Ensure proper VAE precision
    vae_dtype = TORCH_DTYPE_MAPPING.get(args.vae_precision, torch.float32)
    
    # For mixed precision training we cast the text_encoder and vae weights to the appropriate precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move VAE to device with correct precision
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Move text encoders to device with correct precision
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    
    # Initialize null_conditioning and time_ids with correct dtype after setting weight_dtype
    null_conditioning = compute_null_conditioning().to(dtype=weight_dtype)
    add_time_ids = compute_time_ids(weight_dtype)

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    # Freeze text encoder
    text_encoder_1.requires_grad_(False)
    text_encoder_1.eval()  # Set to eval mode

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

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
                    if isinstance(model, FidelityMLP):
                        # Save FidelityMLP
                        fidelity_mlp_path = os.path.join(output_dir, "fidelity_mlp")
                        os.makedirs(fidelity_mlp_path, exist_ok=True)
                        unwrapped_fidelity_mlp = accelerator.unwrap_model(model)
                        unwrapped_fidelity_mlp.save_pretrained(fidelity_mlp_path)
                    else:
                        # Original save logic for other models
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
                # pop models so that they are not loaded again
                model = models.pop()

                # Check if the model is FidelityMLP
                if isinstance(model, FidelityMLP):
                    # Load FidelityMLP
                    fidelity_mlp_path = os.path.join(input_dir, "fidelity_mlp")
                    if os.path.exists(fidelity_mlp_path):
                        load_model = FidelityMLP.from_pretrained(fidelity_mlp_path)
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                else:
                    # load diffusers style into model
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

    # Load or create FidelityMLP
    if args.fidelity_mlp_path:
        if os.path.exists(args.fidelity_mlp_path):
            logger.info(f"Loading FidelityMLP from {args.fidelity_mlp_path}")
            fidelity_mlp = FidelityMLP.from_pretrained(args.fidelity_mlp_path)
        else:
            logger.info(f"Creating new FidelityMLP and will save to {args.fidelity_mlp_path}")
            hidden_size = text_encoder_1.config.hidden_size
            fidelity_mlp = FidelityMLP(hidden_size)
            os.makedirs(args.fidelity_mlp_path, exist_ok=True)
    else:
        logger.info("Creating new FidelityMLP")
        hidden_size = text_encoder_1.config.hidden_size
        fidelity_mlp = FidelityMLP(hidden_size)
    
    # Move fidelity_mlp to device
    fidelity_mlp.to(accelerator.device)


    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    # Optimizer set for unet and fidelity_mlp

    params_to_optimize = [
        {"params": unet.parameters(), "lr": args.learning_rate},
        {"params": fidelity_mlp.parameters(), "lr": args.fidelity_mlp_learning_rate if args.fidelity_mlp_learning_rate else args.learning_rate}
    ]

    optimizer = optimizer_cls(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
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


    if args.dataset_name is not None:
        
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        try:
           dataset = {"train": load_image_dataset(args.train_data_dir)}
        except Exception as e:
            print("Error loading dataset:", e)
            raise
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

    column_names = dataset["train"].column_names
    print("---column_names---", column_names)

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

    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            # Add random rotation of +90 or -90 degrees for 10% of images
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 90 * (2 * torch.randint(0, 2, (1,)).item() - 1)) 
                             if torch.rand(1).item() < 0.1 else x),
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
        prompt_embeds_all, pooled_prompt_embeds_all = compute_embeddings_for_prompts(captions, text_encoders, tokenizers)
        examples["prompt_embeds"] = prompt_embeds_all
        examples["pooled_prompt_embeds"] = pooled_prompt_embeds_all
        
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
        
        prompt_embeds = torch.concat([example["prompt_embeds"] for example in examples], dim=0)
        pooled_prompt_embeds = torch.concat([example["pooled_prompt_embeds"] for example in examples], dim=0)
        
        result = {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
        
        # Extract fidelity values from prompts using regex
        fidelity_values = []
        for i, example in enumerate(examples):
            prompt_text = tokenizers[0].decode(example.get("input_ids", []), skip_special_tokens=True)
            if not prompt_text:  # If input_ids not available, use the embedded prompt
                prompt_text = tokenizers[0].decode(
                    tokenizers[0].encode(example.get("edit_prompt", ""), truncation=True),
                    skip_special_tokens=True
                )
            
            # Extract fidelity value
            import re
            match = re.search(r"f\s*=?\s*(\d+)|f(\d+)", prompt_text, re.IGNORECASE)
            if match:
                f_int = int(match.group(1) if match.group(1) else match.group(2))
                f_int = max(1, min(f_int, 9))
                # Map to normalized range [0.1, 0.9]
                fidelity_val = 0.1 + (f_int - 1) * (0.8 / 8)
            else:
                # Default to medium fidelity if not specified
                fidelity_val = 0.5
            fidelity_values.append(fidelity_val)
        
        result["fidelity"] = torch.tensor(fidelity_values, dtype=torch.float)
        
        return result

    # DataLoader:
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

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, fidelity_mlp = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, fidelity_mlp
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
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
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Train!
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

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
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

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet):
                # Process one batch at a time with aggressive memory cleanup
                
                # Convert images to latent space with consistent dtype
                edited_pixel_values = batch["edited_pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(edited_pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)  # Ensure consistent dtype

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL additional inputs
                encoder_hidden_states = batch["prompt_embeds"].to(dtype=weight_dtype)  # Ensure correct dtype
                add_text_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)  # Ensure correct dtype

                # Get the additional image embedding for conditioning with consistent dtype
                original_pixel_values = batch["original_pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    original_image_embeds = vae.encode(original_pixel_values).latent_dist.sample()
                    original_image_embeds = original_image_embeds.to(dtype=weight_dtype)  # Ensure consistent dtype

                # Ensure null_conditioning has the right dtype
                null_conditioning = null_conditioning.to(dtype=weight_dtype)

                # Conditioning dropout
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images
                    image_mask_dtype = original_image_embeds.dtype  # Already set to weight_dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the original_image_embeds with the noisy_latents
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Fixed: Create time_ids with the correct dtype
                time_ids = compute_time_ids(weight_dtype)
                
                # SDXL additional conditioning with correct dtype
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": time_ids}

                # Process fidelity with FidelityMLP if available
                fidelity_values = batch.get("fidelity")
                if fidelity_mlp is not None and fidelity_values is not None:
                    fidelity_tensor = fidelity_values.unsqueeze(1).to(device=latents.device, dtype=weight_dtype)
                    
                    # Get the correct target dimension from the hidden states, not the pooled embeddings
                    target_dim = encoder_hidden_states.shape[-1]  # This is the dimension we need to match
                    
                    # Generate embedding with the correct target dimension
                    fidelity_embedding = fidelity_mlp(fidelity_tensor, target_dim=target_dim)
                    fidelity_embedding = fidelity_embedding.to(dtype=weight_dtype)
                    
                    # Ensure dimensions match before adding
                    # Reshape fidelity embedding to match the first token dimension exactly
                    fidelity_embedding = fidelity_embedding.view(encoder_hidden_states.shape[0], 1, -1)
                    
                    # Add to the first token embedding safely
                    encoder_hidden_states = encoder_hidden_states.clone()
                    encoder_hidden_states[:, 0:1] = encoder_hidden_states[:, 0:1] + 0.2 * fidelity_embedding

                # Predict the noise residual
                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Compute loss with same dtype
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Process fidelity loss if applicable
                if fidelity_values is not None and global_step % 10 == 0:
                    # Compute fidelity-related loss components
                    with torch.no_grad():
                        # Get the current predicted image (at this noise level)
                        pred_latents = noise_scheduler.step(
                            model_pred.detach(), timesteps[0], noisy_latents, return_dict=False
                        )[0]
                        
                        # Decode to pixel space
                        pred_latents = pred_latents.to(dtype=weight_dtype)
                        pred_images = vae.decode(pred_latents / vae.config.scaling_factor, return_dict=False)[0]
                        
                        # Compute fidelity loss - relates predicted image to input and target based on desired fidelity
                        fidelity_loss = compute_fidelity_loss(
                            pred_images=pred_images,
                            input_images=batch["original_pixel_values"].to(device=pred_images.device, dtype=pred_images.dtype),
                            target_images=batch["edited_pixel_values"].to(device=pred_images.device, dtype=pred_images.dtype),
                            fidelity_values=fidelity_values
                        )
                        
                        # Log the losses
                        if accelerator.is_main_process and global_step % 100 == 0:
                            logger.info(f"Step {global_step}: Main loss: {loss.item():.4f}, Fidelity loss: {fidelity_loss.item():.4f}")
                else:
                    fidelity_loss = torch.tensor(0.0, device=loss.device)
                # Add fidelity loss with increasing weight over time
                fidelity_weight = min(0.1, 0.01 * (global_step / 1000))
                loss = loss + fidelity_weight * fidelity_loss

                # Backpropagation
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)
                train_loss += loss.item() / args.gradient_accumulation_steps
                
                if accelerator.is_main_process:
                    if args.checkpointing_steps is not None and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # Optionally clean up old checkpoints
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.path.join(args.output_dir, "checkpoint-*")
                            checkpoints = sorted(glob.glob(checkpoints), key=lambda x: int(x.split("-")[-1]))
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"Removing old checkpoints: {', '.join(removing_checkpoints)}"
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(removing_checkpoint)

                # Move validation check here, right after global_step is incremented
                if accelerator.is_main_process:
                    if args.val_images_dir is not None and (
                        (global_step == 200) or  # First validation early in training
                        (global_step == 500) or
                        (global_step == 800) or
                        (global_step == 1200) or  # First validation early in training
                        (args.validation_steps > 0 and global_step % args.validation_steps == 0) or  # Regular validation during training
                        (global_step >= args.max_train_steps)  # Final validation
                    ):
                        logger.info(f"Running validation at step {global_step}")
                        if args.use_ema:
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                            
                        # Create the pipeline with the correct fidelity_mlp
                        pipeline = create_validation_pipeline(args, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, vae, unet, fidelity_mlp, 
                                                               accelerator, args.revision, args.variant)
                        
                        if fidelity_mlp is not None:
                            pipeline.fidelity_mlp = fidelity_mlp
                        
                        accelerator.step = global_step
                        
                        log_validation(
                            pipeline,
                            args,
                            accelerator,
                            generator,
                        )

                        if args.use_ema:
                            ema_unet.restore(unet.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.val_images_dir is not None and (
                (global_step == 10) or  # First validation early in training
                (args.validation_steps > 0 and global_step % args.validation_steps == 0) or  # Regular validation during training
                (global_step >= args.max_train_steps)  # Final validation
            ):
                logger.info(f"Running validation at step {global_step}")
                if args.use_ema:
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                    
                pipeline = create_validation_pipeline(args, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, vae, unet, fidelity_mlp, 
                                                       accelerator, args.revision, args.variant)
                
                if fidelity_mlp is not None:
                    pipeline.fidelity_mlp = fidelity_mlp
                
                log_validation(
                    pipeline,
                    args,
                    accelerator,
                    generator,
                )

                if args.use_ema:
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        unet = unwrap_model(unet)
        text_encoder_1 = unwrap_model(text_encoder_1)
        text_encoder_2 = unwrap_model(text_encoder_2)

        # Save the trained models
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        text_encoder_1.save_pretrained(os.path.join(args.output_dir, "text_encoder_1"))
        text_encoder_2.save_pretrained(os.path.join(args.output_dir, "text_encoder_2"))

        # Save the pipeline
        pipeline = create_validation_pipeline(args, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, vae, unet, fidelity_mlp, 
                                               accelerator, args.revision, args.variant)
        
        if fidelity_mlp is not None:
            pipeline.fidelity_mlp = fidelity_mlp
        
        pipeline.save_pretrained(args.output_dir)
        
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message=f"End of training, step: {global_step}",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if (args.val_image_url is not None) and (args.validation_prompt is not None):
            log_validation(
                pipeline,
                args,
                accelerator,
                generator,
            )

        # At the end of training, save the fidelity_mlp
        if args.fidelity_mlp_path:
            if accelerator.is_main_process:
                # Get the fidelity_mlp from the unwrapped model
                unwrapped_fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)
                unwrapped_fidelity_mlp.save_pretrained(args.fidelity_mlp_path)
                logger.info(f"Saved FidelityMLP to {args.fidelity_mlp_path}")

    # Too noisy to be helpful
    if global_step % args.validation_steps == 0:
        test_fidelities = [0.1, 0.5, 0.9]  # Low, medium, high
        with torch.no_grad():
            for f_val in test_fidelities:
                test_tensor = torch.tensor([[f_val]], device=accelerator.device)
                output = fidelity_mlp(test_tensor)
                logger.info(f"Fidelity {f_val} produces embedding with mean: {output.mean().item()}, std: {output.std().item()}")

    accelerator.end_training()



def compute_fidelity_loss(pred_images, input_images, target_images, fidelity_values):
    """
    Compute a loss that teaches the model what fidelity means.
    
    Args:
        pred_images: The predicted images from the model
        input_images: The input drawings/images
        target_images: The target edited images
        fidelity_values: Tensor of fidelity values (normalized to [0,1])
    
    Returns:
        A loss tensor that encourages the model to respect fidelity values
    """
    # batch_size = pred_images.shape[0]
    
    
    # Calculate structural similarity between pred and input
    pred_input_diff = torch.abs(pred_images - input_images).mean(dim=[1,2,3])
    
    # Calculate structural similarity between pred and target
    pred_target_diff = torch.abs(pred_images - target_images).mean(dim=[1,2,3])
    
    # For high fidelity, minimize pred_input_diff
    # For low fidelity, minimize pred_target_diff
    fidelity_weights = fidelity_values.view(-1)
    inverse_fidelity = 1 - fidelity_weights
    
    # Weighted loss based on fidelity
    loss = (fidelity_weights * pred_input_diff) + (inverse_fidelity * pred_target_diff)
    
    return loss.mean()




def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")





if __name__ == "__main__":
    main()

# Add this helper function to pre-compute pipeline for validation
def create_validation_pipeline(args, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, vae, unet, fidelity_mlp, 
                               accelerator, revision, variant):
    """Create a pipeline for validation and cache it to avoid repeated creation."""
    with torch.no_grad():
        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            vae=vae,
            unet=accelerator.unwrap_model(unet) if not isinstance(unet, torch.nn.Module) else unet,
            revision=revision,
            variant=variant,
            torch_dtype=text_encoder_1.dtype,
        )
        
        if fidelity_mlp is not None:
            pipeline.fidelity_mlp = fidelity_mlp
            
    return pipeline
