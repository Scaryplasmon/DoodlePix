import argparse
import logging
import math
import os
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import datasets
import numpy as np
import PIL
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
from transformers import AutoTokenizer, PretrainedConfig
from typing import Optional
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
    StableDiffusionXLInstructPix2PixPipeline,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from fidelity_mlp import FidelityMLP
from PIL import Image as PILImage
from datasets import Dataset, Image
import re
import lmdb
import pickle
import io
from torch.utils.data import Dataset as TorchDataset
import random
import lpips
from pytorch_msssim import ssim
import json

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("file_name", "edited_image", "edit_prompt"),
}
WANDB_TABLE_COL_NAMES = ["file_name", "edited_image", "edit_prompt"]
TORCH_DTYPE_MAPPING = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# Replace the existing log_validation function with this one

def log_validation(
    pipeline,  # Expects pipeline with fidelity_mlp already attached
    args,
    accelerator,
    generator,
    global_step,
    is_final_validation=False,
):
    logger.info("Running validation...")

    if args.val_images_dir is None:
        logger.warning("`args.val_images_dir` not set. Skipping detailed image validation.")
        return

    validation_dir = os.path.join(args.output_dir, "validation_images")
    step_dir = os.path.join(validation_dir, f"step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    logger.info(f"Original pipeline VAE dtype: {pipeline.vae.dtype}")
    pipeline.vae = pipeline.vae.to(dtype=torch.float32)
    logger.info(f"Set pipeline VAE dtype for validation: {pipeline.vae.dtype}")

    # Get all validation images and their corresponding prompts
    val_images_dir = args.val_images_dir
    try:
        image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error accessing validation directory '{val_images_dir}': {e}")
        return

    if not image_files:
        logger.warning(f"No validation images found in '{val_images_dir}'. Skipping detailed image validation.")
        return

    logger.info(f"Found {len(image_files)} validation images in '{val_images_dir}'")

    edited_images_log = [] # Store data for logging

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        # Use autocast only if mixed precision is enabled
        autocast_ctx = torch.autocast(accelerator.device.type) if accelerator.mixed_precision != "no" else nullcontext()

    fidelity_levels = [2, 8]  # Example: Low, high fidelity

    with autocast_ctx:
        for image_file in tqdm(image_files, desc="Validation Images"):
            try:
                base_name = os.path.splitext(image_file)[0]
                image_path = os.path.join(val_images_dir, image_file)
                prompt_path = os.path.join(val_images_dir, f"{base_name}.txt")

                if not os.path.exists(prompt_path):
                    logger.warning(f"Skipping {image_file} - no corresponding prompt file '{base_name}.txt' found")
                    continue

                with open(prompt_path, 'r', encoding='utf-8') as f:
                    validation_prompt_base = f.read().strip()

                original_image = PILImage.open(image_path).convert("RGB")
                original_image_resized = original_image.resize((args.resolution, args.resolution))

                # Save resized original image once per step
                try:
                    original_save_path = os.path.join(step_dir, f"original_{base_name}.png")
                    if not os.path.exists(original_save_path): # Avoid saving duplicates if called multiple times
                         original_image_resized.save(original_save_path)
                except Exception as e:
                    logger.error(f"Failed to save original image {base_name}: {e}")
                    continue

                current_edited_images = []
                current_fidelity_prompts = []
                for fidelity in fidelity_levels:
                    # Create a new generator for each image/fidelity pair for determinism if needed
                    val_seed = args.seed if args.seed is not None else np.random.randint(2**32)
                    generator = torch.Generator(device=accelerator.device).manual_seed(val_seed)

                    # Prepend fidelity prefix to the base prompt
                    fidelity_prompt = f"f{fidelity},{validation_prompt_base}"
                    current_fidelity_prompts.append(fidelity_prompt)

                    try:
                        # SDXL pipeline call
                        edited_image = pipeline(
                            prompt=fidelity_prompt,
                            image=original_image, # Pass original non-resized image
                            height=args.resolution,
                            width=args.resolution,
                            num_inference_steps=26, # Use appropriate steps for SDXL
                            image_guidance_scale=1.5,
                            guidance_scale=5.0, # Use appropriate guidance for SDXL
                            generator=generator,
                            # Other SDXL specific args like original_size, target_size could be added if needed
                        ).images[0]

                        edited_image.save(os.path.join(step_dir, f"edited_{base_name}_f{fidelity}.png"))
                        current_edited_images.append(edited_image)

                        # # Save the exact prompt used
                        # with open(os.path.join(step_dir, f"prompt_{base_name}_f{fidelity}.txt"), "w", encoding='utf-8') as f:
                        #     f.write(fidelity_prompt)

                    except Exception as e:
                        logger.error(f"Failed to generate/save edited image {base_name} with fidelity {fidelity}: {e}", exc_info=True)
                        # Add a placeholder or skip logging this image
                        current_edited_images.append(None) # Mark failure
                        continue

                # Add results for this image to the log list
                if any(img is not None for img in current_edited_images): # Log if at least one succeeded
                    edited_images_log.append({
                         "base_name": base_name,
                         "original": original_image_resized,
                         "edited": current_edited_images, # List contains PIL images or None
                         "prompt": validation_prompt_base,
                         "fidelity_levels": fidelity_levels
                     })

            except Exception as e:
                logger.error(f"Failed to process validation image {image_file}: {e}", exc_info=True)
                continue

    # Log to WandB if enabled
    if accelerator.is_main_process:
        try:
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=["Image Name", "Original Image", "Edited Image", "Base Prompt", "Fidelity"])
                    for data in edited_images_log:
                        for i, edited_image in enumerate(data["edited"]):
                            fidelity = data["fidelity_levels"][i]
                            if edited_image is not None: # Only log successful generations
                                wandb_table.add_data(
                                    data["base_name"],
                                    wandb.Image(data["original"]),
                                    wandb.Image(edited_image),
                                    data["prompt"],
                                    f"f{fidelity}"
                                )
                    logger_name = "final_validation" if is_final_validation else "validation"
                    tracker.log({logger_name: wandb_table}, step=global_step)
                    logger.info(f"Logged {len(edited_images_log)} validation sets to WandB.")
        except Exception as e:
            logger.error(f"Failed to log validation results to WandB: {e}", exc_info=True)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train Stable Diffusion XL for InstructPix2Pix.")
    parser.add_argument(
        "--no_proxy",
        default=False,
        action="store_true",
        help=(
            "Whether to use proxy loss for fidelity or actual fidelity loss"
        ),
    )
    parser.add_argument(
        "--fidelity_loss_type",
        type=str,
        default="l1",
        choices=["l1", "ssim", "lpips"],
        help="Type of loss for the full fidelity calculation ('l1', 'ssim', 'lpips'). Only used if --no_proxy is True.",
    )
    parser.add_argument(
        "--use_fidelity_loss",
        action="store_true",
        help="Enable the explicit fidelity loss term during training.",
    )
    parser.add_argument(
        "--fidelity_weight",
        type=float,
        default=0.1, # Start with a small weight, requires tuning
        help="Weighting factor for the explicit fidelity loss term.",
    )
    parser.add_argument(
        "--fidelity_loss_freq",
        type=int,
        default=10, # Calculate every step by default
        help="Calculate the fidelity loss every N steps (set > 1 to reduce computational cost).",
    )
    parser.add_argument(
        "--fidelity_mlp_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the fidelity MLP. Defaults to same as UNet if not set.",
    )
    parser.add_argument(
        "--fidelity_mlp_path",
        type=str,
        default=None,
        help="Path to a pre-trained fidelity MLP model or directory to save/load the trained MLP.",
    )
    parser.add_argument(
         "--val_images_dir",
         type=str,
         default=None,
         help="Directory containing validation images and corresponding .txt prompt files for detailed validation.",
    )
    parser.add_argument(
         "--validation_steps",
         type=int,
         default=1000, 
         help=(
             "Run validation every X steps. Set to a large number or -1 to disable during training."
             " Also runs validation at specific early steps (e.g., 200, 500) and at the end."
         ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--vae_precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp16",
        help=(
            "The vanilla SDXL 1.0 VAE can cause NaNs due to large activation values. Some custom models might already have a solution"
            " to this problem, and this flag allows you to use mixed precision to stabilize training."
        ),
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
        "--val_image_url_or_path",
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
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
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
        default="fp16",
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
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


class LMDBImagePromptDataset(TorchDataset):
    """
    PyTorch Dataset for loading image pairs and prompts from an LMDB database.
    """
    def __init__(self, lmdb_path, transform=None):
        """
        Args:
            lmdb_path (str): Path to the LMDB database directory.
            transform (callable, optional): Optional transform to be applied
                                            on a sample. Should handle a dict
                                            {'input_image': PIL, 'edited_image': PIL, 'edit_prompt': str}
        """
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None # Environment will be opened in __getitem__ if needed, better for multiprocessing

        logger.info(f"Initializing LMDB dataset from: {lmdb_path}")

        try:
            # Open temporarily to get keys and length
            temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            with temp_env.begin(write=False) as txn:
                self.length = txn.stat()['entries']
                logger.info("Reading keys from LMDB... (This might take a moment)")
                # Read all keys into memory - requires keys fit in RAM, usually okay.
                self.keys = [key for key in tqdm(txn.cursor().iternext(keys=True, values=False), total=self.length, desc="Loading LMDB keys")]
            temp_env.close()
            logger.info(f"Found {self.length} entries in LMDB.")
        except lmdb.Error as e:
            logger.error(f"Failed to open or read LMDB at {lmdb_path}: {e}")
            raise IOError(f"Could not initialize LMDB dataset at {lmdb_path}")

    def _init_db(self):
        # Open the environment in the current process.
        # lock=False is important for multi-worker DataLoader
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Open DB in each worker process if not already opened
        if self.env is None:
            self._init_db()

        key = self.keys[index]
        with self.env.begin(write=False) as txn:
            serialized_data = txn.get(key)

        if serialized_data is None:
             raise KeyError(f"Key {key.decode('utf-8')} not found in LMDB (index {index}). This shouldn't happen.")

        # Deserialize data
        try:
            input_img_bytes, edited_img_bytes, edit_prompt = pickle.loads(serialized_data)
        except pickle.UnpicklingError as e:
            raise RuntimeError(f"Failed to unpickle data for key {key.decode('utf-8')}: {e}")

        # Decode images from bytes using PIL
        try:
            input_image = PILImage.open(io.BytesIO(input_img_bytes)).convert('RGB')
            edited_image = PILImage.open(io.BytesIO(edited_img_bytes)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to decode image for key {key.decode('utf-8')}: {e}")

        # Prepare sample dictionary
        sample = {
            'input_image': input_image,
            'edited_image': edited_image,
            'edit_prompt': edit_prompt # Prompt is already a string
        }

        # Apply transforms (which should handle the dictionary)
        if self.transform:
            # Your preprocess_train function needs to be adapted to take this dict
            # or you define a transform chain that works on PIL images directly.
            # Let's assume preprocess_train can be adapted or replaced by a transform.
             sample = self.transform(sample)

        return sample

def compute_fidelity_loss(
    pred_images: torch.Tensor,
    input_images: torch.Tensor,
    target_images: torch.Tensor,
    fidelity_values: torch.Tensor,
    loss_type: str = "l1",
    lpips_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu") # Pass the device
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
    # Ensure images are on the correct device and float32
    pred_images = pred_images.to(device=device, dtype=torch.float32)
    input_images = input_images.to(device=device, dtype=torch.float32)
    target_images = target_images.to(device=device, dtype=torch.float32)
    fidelity_values = fidelity_values.to(device=device, dtype=torch.float32).view(-1) # Ensure shape [B]

    batch_size = pred_images.shape[0]
    dist_to_input = torch.zeros(batch_size, device=device)
    dist_to_target = torch.zeros(batch_size, device=device)

    if loss_type == "l1":
        # L1 Loss (Absolute Difference)
        dist_to_input = torch.abs(pred_images - input_images).mean(dim=[1, 2, 3])
        dist_to_target = torch.abs(pred_images - target_images).mean(dim=[1, 2, 3])

    elif loss_type == "ssim":
        # SSIM Loss: ssim function returns similarity (higher is better), so loss is 1 - ssim
        # Ensure images are in [0, 1] range for SSIM if they are not already
        pred_images_01 = (pred_images + 1) / 2 if pred_images.min() < 0 else pred_images
        input_images_01 = (input_images + 1) / 2 if input_images.min() < 0 else input_images
        target_images_01 = (target_images + 1) / 2 if target_images.min() < 0 else target_images

        # Calculate SSIM per image in the batch
        for i in range(batch_size):
             # data_range is 1.0 since images are normalized to [0, 1]
            dist_to_input[i] = 1.0 - ssim(pred_images_01[i].unsqueeze(0), input_images_01[i].unsqueeze(0), data_range=1.0, size_average=True)
            dist_to_target[i] = 1.0 - ssim(pred_images_01[i].unsqueeze(0), target_images_01[i].unsqueeze(0), data_range=1.0, size_average=True)

    elif loss_type == "lpips":
        # LPIPS Loss: Measures perceptual distance (lower is better)
        if lpips_model is None:
            raise ValueError("lpips_model must be provided for loss_type 'lpips'")
        print("the device is", device)
        # LPIPS expects input range [-1, 1]
        pred_images_norm = pred_images if pred_images.min() >= -1 else (pred_images * 2) - 1
        input_images_norm = input_images if input_images.min() >= -1 else (input_images * 2) - 1
        target_images_norm = target_images if target_images.min() >= -1 else (target_images * 2) - 1

        dist_to_input = lpips_model(pred_images_norm, input_images_norm).squeeze()
        dist_to_target = lpips_model(pred_images_norm, target_images_norm).squeeze()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Weighted combination based on fidelity:
    # High fidelity (close to 1) -> emphasize distance to input
    # Low fidelity (close to 0) -> emphasize distance to target
    weighted_loss = (fidelity_values * dist_to_input) + ((1 - fidelity_values) * dist_to_target)

    return weighted_loss.mean()

def save_base_model_structure(output_dir, tokenizers, text_encoders, vae, scheduler, pipeline_class_name="StableDiffusionXLInstructPix2PixPipeline"):
    """
    Saves the base structure (tokenizers, text encoders, VAE, scheduler, configs)
    of an InstructPix2Pix XL model to a specified directory, excluding UNet and Fidelity MLP.
    """
    base_structure_dir = os.path.join(output_dir, "base_model_structure")
    logger.info(f"Saving base model structure (excluding UNet/MLP) to: {base_structure_dir}")
    os.makedirs(base_structure_dir, exist_ok=True)

    components = {
        "tokenizer": tokenizers[0],
        "tokenizer_2": tokenizers[1],
        "text_encoder": text_encoders[0],
        "text_encoder_2": text_encoders[1],
        "vae": vae,
        "scheduler": scheduler,
    }

    model_index = {
        "_class_name": pipeline_class_name,
        "_diffusers_version": diffusers.__version__,
    }

    for name, component in components.items():
        try:
            save_path = os.path.join(base_structure_dir, name)
            component.save_pretrained(save_path)
            logger.info(f"Saved {name} to {save_path}")

            # Add component info to model_index.json
            # Determine library (diffusers or transformers)
            library_name = "diffusers" if "diffusers" in component.__class__.__module__ else "transformers"
            model_index[name] = [library_name, component.__class__.__name__]

        except Exception as e:
            logger.error(f"Failed to save component '{name}': {e}")
            # Decide if you want to raise error or just log and continue
            # raise e # Uncomment to stop if any component fails saving

    # Write model_index.json (excluding unet and fidelity_mlp)
    try:
        model_index_path = os.path.join(base_structure_dir, "model_index.json")
        with open(model_index_path, "w", encoding="utf-8") as f:
            json.dump(model_index, f, indent=2)
        logger.info(f"Saved model_index.json to {model_index_path}")
    except Exception as e:
        logger.error(f"Failed to save model_index.json: {e}")

    logger.info("Base model structure saving complete.")

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

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

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

    # ----- START LOAD/INIT FIDELITY MLP -----
    fidelity_mlp = None
    if args.fidelity_mlp_path:
        if os.path.exists(args.fidelity_mlp_path) and os.path.isdir(args.fidelity_mlp_path):
             # Try loading from directory
             try:
                 logger.info(f"Loading FidelityMLP from directory: {args.fidelity_mlp_path}")
                 fidelity_mlp = FidelityMLP.from_pretrained(args.fidelity_mlp_path)
             except Exception as e:
                 logger.error(f"Failed to load FidelityMLP from {args.fidelity_mlp_path}: {e}. Creating a new one.")
                 fidelity_mlp = None # Reset to ensure creation below
        elif os.path.exists(args.fidelity_mlp_path) and not os.path.isdir(args.fidelity_mlp_path):
             logger.warning(f"Fidelity MLP path {args.fidelity_mlp_path} exists but is not a directory. Cannot load. Creating a new one.")
             fidelity_mlp = None # Reset to ensure creation below


    if fidelity_mlp is None:
        # Determine hidden size for initialization
        # Load text_encoder_2 config temporarily just to get hidden size
        try:
             temp_config = PretrainedConfig.from_pretrained(
                 args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
             )
             hidden_size = temp_config.hidden_size
             del temp_config
        except Exception:
            # Fallback if text_encoder_2 doesn't exist or fails
            try:
                temp_config = PretrainedConfig.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
                )
                hidden_size = temp_config.hidden_size
                del temp_config
                logger.warning("Using text_encoder_1 hidden size for FidelityMLP init.")
            except Exception as e:
                hidden_size = 1024 # SDXL Large Encoder size as a reasonable default
                logger.error(f"Could not determine text encoder hidden size due to error: {e}. Defaulting FidelityMLP hidden size to {hidden_size}.")

        logger.info(f"Creating new FidelityMLP with hidden size {hidden_size}.")
        fidelity_mlp = FidelityMLP(hidden_size)
        # If path was specified but didn't exist, create directory structure for saving later
        if args.fidelity_mlp_path and not os.path.exists(args.fidelity_mlp_path):
             try:
                 os.makedirs(args.fidelity_mlp_path, exist_ok=True)
                 logger.info(f"Directory created for saving new FidelityMLP: {args.fidelity_mlp_path}")
             except OSError as e:
                 logger.error(f"Could not create directory {args.fidelity_mlp_path}: {e}. MLP will not be saved unless path is corrected.")
                 args.fidelity_mlp_path = None # Prevent saving errors later

    fidelity_mlp.to(accelerator.device) # Move to device early
    # ----- END LOAD/INIT FIDELITY MLP -----

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the XL InstructPix2Pix UNet from the pretrained UNet.")
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

    # Create EMA for the unet.
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
        def save_model_hook(models, weights, output_dir):
            unet_model = None
            fidelity_mlp_model = None
            unet_weights = None
            fidelity_mlp_weights = None

            # Separate models and weights
            temp_weights = list(weights) # Copy weights list
            saved_indices = []

            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), UNet2DConditionModel):
                     unet_model = model
                     unet_weights = temp_weights[i]
                     saved_indices.append(i)
                elif isinstance(accelerator.unwrap_model(model), FidelityMLP):
                     fidelity_mlp_model = model
                     fidelity_mlp_weights = temp_weights[i]
                     saved_indices.append(i)

            # Remove saved weights from original list by index (descending order)
            for i in sorted(saved_indices, reverse=True):
                 weights.pop(i)

            if accelerator.is_main_process:
                # Save UNet
                if unet_model is not None:
                    unet_save_path = os.path.join(output_dir, "unet")
                    if args.use_ema:
                        # Save EMA UNet state if available and different from current UNet
                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    unet_model.save_pretrained(unet_save_path) # Save current UNet state
                    logger.info(f"UNet saved to {unet_save_path}")

                # Save Fidelity MLP
                if fidelity_mlp_model is not None:
                    fidelity_mlp_save_path = os.path.join(output_dir, "fidelity_mlp")
                    # Always save the current state of fidelity_mlp
                    unwrapped_fidelity_mlp = accelerator.unwrap_model(fidelity_mlp_model)
                    unwrapped_fidelity_mlp.save_pretrained(fidelity_mlp_save_path)
                    logger.info(f"FidelityMLP saved to {fidelity_mlp_save_path}")


        def load_model_hook(models, input_dir):
            unet_model_idx = -1
            fidelity_mlp_model_idx = -1

            # Find models by type
            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), UNet2DConditionModel):
                     unet_model_idx = i
                elif isinstance(accelerator.unwrap_model(model), FidelityMLP):
                     fidelity_mlp_model_idx = i

            # Load UNet (and EMA if used)
            if unet_model_idx != -1:
                unet_model = models[unet_model_idx]
                if args.use_ema:
                     ema_path = os.path.join(input_dir, "unet_ema")
                     if os.path.exists(ema_path):
                         try:
                             load_ema_model = EMAModel.from_pretrained(ema_path, UNet2DConditionModel)
                             ema_unet.load_state_dict(load_ema_model.state_dict())
                             ema_unet.to(accelerator.device)
                             logger.info(f"Loaded EMA UNet state from {ema_path}")
                             del load_ema_model
                         except Exception as e:
                             logger.error(f"Failed to load EMA UNet state from {ema_path}: {e}")
                     else:
                         logger.warning(f"EMA UNet state not found at {ema_path}, EMA weights not loaded.")

                # Load main UNet weights
                unet_path = os.path.join(input_dir, "unet")
                if os.path.exists(unet_path):
                    try:
                        load_unet_model = UNet2DConditionModel.from_pretrained(unet_path)
                        unet_model.register_to_config(**load_unet_model.config)
                        unet_model.load_state_dict(load_unet_model.state_dict())
                        logger.info(f"Loaded UNet state from {unet_path}")
                        del load_unet_model
                    except Exception as e:
                        logger.error(f"Failed to load UNet state from {unet_path}: {e}")
                else:
                    logger.error(f"UNet state directory not found at {unet_path}, UNet not loaded.")


            # Load Fidelity MLP
            if fidelity_mlp_model_idx != -1:
                fidelity_mlp_model = models[fidelity_mlp_model_idx]
                fidelity_mlp_path = os.path.join(input_dir, "fidelity_mlp")
                if os.path.exists(fidelity_mlp_path):
                     try:
                         load_fidelity_model = FidelityMLP.from_pretrained(fidelity_mlp_path)
                         fidelity_mlp_model.load_state_dict(load_fidelity_model.state_dict())
                         logger.info(f"Loaded FidelityMLP state from {fidelity_mlp_path}")
                         del load_fidelity_model
                     except Exception as e:
                         logger.error(f"Failed to load FidelityMLP state from {fidelity_mlp_path}: {e}")
                else:
                    logger.warning(f"FidelityMLP state directory not found at {fidelity_mlp_path}, MLP not loaded.")

            # Pop models so Accelerate doesn't try default loading
            # Important: Pop in reverse order of index to avoid messing up indices
            indices_to_pop = sorted([idx for idx in [unet_model_idx, fidelity_mlp_model_idx] if idx != -1], reverse=True)
            for i in indices_to_pop:
                models.pop(i)


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

    # Create parameter groups for different learning rates
    unet_lr = args.learning_rate
    fidelity_mlp_lr = args.fidelity_mlp_learning_rate if args.fidelity_mlp_learning_rate is not None else unet_lr

    params_to_optimize = [
         {"params": unet.parameters(), "lr": unet_lr},
         {"params": fidelity_mlp.parameters(), "lr": fidelity_mlp_lr}
    ]
    logger.info(f"Optimizer settings: UNet LR={unet_lr}, FidelityMLP LR={fidelity_mlp_lr}")

    optimizer = optimizer_cls(
        params_to_optimize, # Use the list of param groups
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    def load_image_dataset(source_dir):
        """Loads images and prompts from a directory structure."""
        logger.info(f"Loading image dataset from: {source_dir}")
        input_dir = os.path.join(source_dir, "input_image")
        edited_dir = os.path.join(source_dir, "edited_image")
        prompt_dir = os.path.join(source_dir, "edit_prompt")

        if not os.path.isdir(input_dir) or not os.path.isdir(edited_dir) or not os.path.isdir(prompt_dir):
            raise FileNotFoundError(f"Dataset directories not found or not structured correctly in {source_dir}. "
                                "Expected 'input_image', 'edited_image', 'edit_prompt' subdirectories.")

        try:
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            logger.info(f"Found {len(image_files)} input images.")
        except Exception as e:
            logger.error(f"Error listing files in {input_dir}: {e}")
            raise

        dataset_dict = {
            "input_image": [],
            "edited_image": [],
            "edit_prompt": []
        }
        missing_files_count = 0

        for img_file in tqdm(image_files, desc="Loading dataset items"):
            base_name = os.path.splitext(img_file)[0]

            input_img_path = os.path.join(input_dir, img_file)
            edited_img_path = os.path.join(edited_dir, img_file) # Assume same filename for edited image
            prompt_path = os.path.join(prompt_dir, f"{base_name}.txt")

            # Check if all corresponding files exist
            if not os.path.exists(edited_img_path):
                missing_files_count += 1
                continue
            if not os.path.exists(prompt_path):
                missing_files_count += 1
                continue

            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    edit_prompt = f.read().strip()
                    # Ensure prompt includes fidelity prefix "fX,"
                    if not re.match(r"^\s*f\s*\d+,?\s*", edit_prompt, re.IGNORECASE):
                        missing_files_count += 1
                        continue

                # Check if images can be opened (basic check)
                try:
                    PILImage.open(input_img_path).close()
                    PILImage.open(edited_img_path).close()
                except Exception as img_err:
                    missing_files_count += 1
                    continue

                dataset_dict["input_image"].append(input_img_path)
                dataset_dict["edited_image"].append(edited_img_path)
                dataset_dict["edit_prompt"].append(edit_prompt)

            except Exception as e:
                logger.warning(f"Error processing item {base_name}: {e}")
                missing_files_count += 1
                continue

        if missing_files_count > 0:
            logger.warning(f"Skipped {missing_files_count} items due to missing files, invalid prompts, or image errors.")
        if not dataset_dict["input_image"]:
            raise ValueError(f"No valid data items found in {source_dir}. Check file structure, prompt format (must start with 'fX,'), and image validity.")


        logger.info(f"Successfully loaded {len(dataset_dict['input_image'])} valid items.")

        # Create Hugging Face Dataset object
        hf_dataset = Dataset.from_dict(dataset_dict)
        # Cast image columns to Image type
        hf_dataset = hf_dataset.cast_column("input_image", Image())
        hf_dataset = hf_dataset.cast_column("edited_image", Image())

        return hf_dataset

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        # Rename columns for consistency if using standard dataset
        if args.dataset_name in DATASET_NAME_MAPPING:
             dataset = dataset.rename_columns({
                  DATASET_NAME_MAPPING[args.dataset_name][0]: "input_image",
                  DATASET_NAME_MAPPING[args.dataset_name][1]: "edited_image",
                  DATASET_NAME_MAPPING[args.dataset_name][2]: "edit_prompt",
             })
    elif args.train_data_dir is not None:
        try:
            logger.info(f"Loading dataset using LMDB from: {args.train_data_dir}")
            
            # Define image transformations including the SAME augmentations from preprocess_train
            train_image_transforms = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.RandomChoice([
                    transforms.Lambda(lambda x: x),  # No rotation (0 degrees)
                    transforms.RandomRotation((90, 90)),  # +90 degrees exactly
                    transforms.RandomRotation((-90, -90)),  # -90 degrees exactly
                ]),
                transforms.ToTensor(),  # Convert PIL to Tensor [0.0, 1.0]
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1.0, 1.0]
            ])
            
            # Transform function to apply to samples from LMDB
            def transform_fn(sample):
                # Apply SAME random transform to both images to maintain correspondence
                seed = torch.randint(0, 2147483647, (1,)).item()
                
                torch.manual_seed(seed)
                random.seed(seed)
                input_tensor = train_image_transforms(sample['input_image'])
                
                torch.manual_seed(seed)
                random.seed(seed)
                edited_tensor = train_image_transforms(sample['edited_image'])
                
                return {
                    "original_pixel_values": input_tensor,
                    "edited_pixel_values": edited_tensor,
                    "captions": sample['edit_prompt']
                }
            
            # Create LMDB dataset
            lmdb_dataset = LMDBImagePromptDataset(lmdb_path=args.train_data_dir, transform=transform_fn)
            
            # Add column_names attribute to make compatible with HF datasets
            column_names = ["input_image", "edited_image", "edit_prompt"]
            lmdb_dataset.column_names = column_names
            
            # Truncate dataset if needed
            if args.max_train_samples is not None and args.max_train_samples < len(lmdb_dataset):
                logger.info(f"Truncating dataset to first {args.max_train_samples} samples")
                from torch.utils.data import Subset
                indices = list(range(min(args.max_train_samples, len(lmdb_dataset))))
                lmdb_dataset_subset = Subset(lmdb_dataset, indices)
                # Preserve column_names attribute on the subset
                lmdb_dataset_subset.column_names = column_names
                lmdb_dataset = lmdb_dataset_subset
            
            # Create a dict with train split to match expected format
            dataset = {"train": lmdb_dataset}
            
            # Set column values directly to avoid needing to access dataset["train"].column_names later
            original_image_column = "input_image"
            edited_image_column = "edited_image"  
            edit_prompt_column = "edit_prompt"
            
            # Define custom collate_fn for DataLoader
            def collate_fn(examples):
                batch = {}
                batch["original_pixel_values"] = torch.stack([example["original_pixel_values"] for example in examples])
                batch["edited_pixel_values"] = torch.stack([example["edited_pixel_values"] for example in examples])
                batch["captions"] = [example["captions"] for example in examples]
                return batch
            
            # Skip the dataset["train"].with_transform() steps since we're using a different dataset class
            train_dataset = dataset["train"]
            
            # Create DataLoader directly
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=args.train_batch_size,
                num_workers=args.dataloader_num_workers,
                pin_memory=True  # Better performance with CUDA
            )
            
            logger.info(f"Successfully loaded LMDB dataset with {len(train_dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to load LMDB dataset from {args.train_data_dir}: {e}", exc_info=True)
            raise
    else:
        raise ValueError("Either --dataset_name or --train_data_dir must be specified.")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
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

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        warnings.warn(f"weight_dtype {weight_dtype} may cause nan during vae encoding", UserWarning)

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        warnings.warn(f"weight_dtype {weight_dtype} may cause nan during vae encoding", UserWarning)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions, tokenizer):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
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
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    # Load scheduler, tokenizer and models.
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
    text_encoder_cls_1 = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = text_encoder_cls_1.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_2 = text_encoder_cls_2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )

    # We ALWAYS pre-compute the additional condition embeddings needed for SDXL
    # UNet as the model is already big and it uses two text encoders.
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Set UNet to trainable.
    unet.train()
    
    # if accelerator.is_main_process:
    #     save_base_model_structure(
    #         output_dir=args.output_dir,
    #         tokenizers=[tokenizer_1, tokenizer_2],
    #         text_encoders=[text_encoder_1, text_encoder_2],
    #         vae=vae,
    #         scheduler=noise_scheduler,
    #         # Optional: pass the actual pipeline class if needed, otherwise default works
    #         # pipeline_class_name=StableDiffusionXLInstructPix2PixPipeline.__name__
    #     )

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
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
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompts(text_encoders, tokenizers, prompts):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []

        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)

        return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)

    # Adapted from examples.dreambooth.train_dreambooth_lora_sdxl
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(text_encoders, tokenizers, prompts)
            add_text_embeds_all = pooled_prompt_embeds_all

            prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
            add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
        return prompt_embeds_all, add_text_embeds_all

    # Get null conditioning
    def compute_null_conditioning():
        null_conditioning_list = []
        for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
            null_conditioning_list.append(
                a_text_encoder(
                    tokenize_captions([""], tokenizer=a_tokenizer).to(accelerator.device),
                    output_hidden_states=True,
                ).hidden_states[-2]
            )
        return torch.concat(null_conditioning_list, dim=-1)

    null_conditioning = compute_null_conditioning()

    def compute_time_ids():
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        original_size = target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)

    add_time_ids = compute_time_ids()

    def preprocess_train(examples):
        processed = {}
        preprocessed_images = preprocess_images(examples) # uses global original_image_column etc.
        original_images, edited_images = preprocessed_images
        processed["original_pixel_values"] = original_images.reshape(-1, 3, args.resolution, args.resolution)
        processed["edited_pixel_values"] = edited_images.reshape(-1, 3, args.resolution, args.resolution)
        processed["captions"] = list(examples[edit_prompt_column]) # List of strings
        return processed

    with accelerator.main_process_first():
        if args.max_train_samples is not None and args.dataset_name is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        
        # Only apply with_transform to HuggingFace datasets (not LMDB)
        if not isinstance(dataset["train"], LMDBImagePromptDataset) and not hasattr(dataset["train"], "_dataset") and not isinstance(getattr(dataset["train"], "_dataset", None), LMDBImagePromptDataset):
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)
        else:
            # For LMDB dataset (transformations already applied in transform_fn)
            train_dataset = dataset["train"]

    def collate_fn(examples):
         batch = {}
         batch["original_pixel_values"] = torch.stack([example["original_pixel_values"] for example in examples]).contiguous().float()
         batch["edited_pixel_values"] = torch.stack([example["edited_pixel_values"] for example in examples]).contiguous().float()
         batch["captions"] = [example["captions"] for example in examples] # List of strings, shape [batch_size]
         return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, fidelity_mlp = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, fidelity_mlp
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=TORCH_DTYPE_MAPPING[args.vae_precision])

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix-xl", config=vars(args))

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
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Accumulate UNet AND FidelityMLP gradients
            with accelerator.accumulate(unet):
                # Cast input pixel values to the VAE's expected dtype (fp32)
                edited_pixel_values_vae = batch["edited_pixel_values"].to(dtype=vae.dtype)
                original_pixel_values_vae = batch["original_pixel_values"].to(dtype=vae.dtype)

                # Encode images using VAE (output latents will be float32)
                latents = vae.encode(edited_pixel_values_vae).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise (will be float32, matching latents)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise (result noisy_latents is float32)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ----- START ENCODE PROMPT + FIDELITY INJECTION -----
                captions = batch["captions"] # List of strings
                bsz = len(captions)

                clean_captions_batch = []
                fidelity_values_batch = []
                default_fidelity = 0.7


                # 1. Extract Fidelity and Clean Captions
                for caption in captions:
                    # Regex to find "f[number]," at the start, ignoring spaces
                    match = re.match(r"^\s*f\s*(\d+),?\s*", caption, re.IGNORECASE)
                    if match:
                        f_int = int(match.group(1))
                        f_int = max(1, min(f_int, 9)) # Clamp between 1 and 9
                        # Normalize to [0.1, 0.9] range
                        fidelity_val = 0.1 + (f_int - 1) * (0.8 / 8)
                        clean_caption = caption[match.end():].strip() # Get text after "fX,"
                    else:
                        fidelity_val = default_fidelity # Default if no prefix found
                        clean_caption = caption # Use original caption
                    clean_captions_batch.append(clean_caption)
                    fidelity_values_batch.append(fidelity_val)
                    
                    
                current_fidelity_mlp = accelerator.unwrap_model(fidelity_mlp) # Get unwrapped MLP

                fidelity_tensor = torch.tensor(fidelity_values_batch, device=accelerator.device).view(-1, 1).to(dtype=current_fidelity_mlp.net[0].weight.dtype)
                
                
                # --- 2. Get Fidelity Embedding from MLP ---
                # Use the correct dtype for the MLP input (often float32 is safer)
                fidelity_tensor = torch.tensor(fidelity_values_batch, device=accelerator.device).view(-1, 1).to(dtype=torch.float32)
                
                # Get the unwrapped MLP model to call forward
                current_fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)
                
                # Generate the base fidelity embedding (shape: [B, mlp_hidden_size])
                # Ensure MLP is in eval mode if it has dropout/batchnorm layers, although yours doesn't seem to.
                # fidelity_mlp.eval() # Optional: uncomment if MLP has dropout/batchnorm
                with torch.no_grad(): # MLP doesn't need gradients for this forward pass
                    base_fidelity_embedding = current_fidelity_mlp(fidelity_tensor)
                # fidelity_mlp.train() # Optional: uncomment if you set it to eval() before

                # Save the normalized fidelity values for loss calculation later (shape: [B])
                fidelity_tensor_for_loss = fidelity_tensor.squeeze(-1)

                # 2. Encode Cleaned Captions and Inject Fidelity
                prompt_embeds_list = []
                pooled_prompt_embeds = None
                injection_scale = args.fidelity_weight
                with torch.no_grad(): # Text encoders are frozen
                    for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders)):
                        text_inputs = tokenizer(
                            clean_captions_batch, # Use cleaned captions
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_input_ids = text_inputs.input_ids.to(text_encoder.device)

                        # Encode with text encoder
                        prompt_embeds_output = text_encoder(text_input_ids, output_hidden_states=True)

                        # Get pooled output from second encoder
                        if idx == 1:
                            if hasattr(prompt_embeds_output, 'pooler_output') and prompt_embeds_output.pooler_output is not None:
                                pooled_prompt_embeds = prompt_embeds_output.pooler_output
                            elif prompt_embeds_output[0].ndim == 2:
                                pooled_prompt_embeds = prompt_embeds_output[0]

                        current_prompt_embeds = prompt_embeds_output.hidden_states[-2]

                        # # Inject Fidelity Embedding (using unwrapped MLP)
                        target_dim = current_prompt_embeds.shape[-1]
                        # Re-run MLP forward pass *if* target_dim is different from MLP's native output
                        # Or, if MLP always outputs fixed size matching one encoder, project/slice here.
                        # Assuming FidelityMLP's _adjust_dimension handles it:
                        fidelity_embedding_adjusted = current_fidelity_mlp(fidelity_tensor, target_dim=target_dim)
                        
                        # Ensure dtype matches before adding
                        fidelity_embedding_adjusted = fidelity_embedding_adjusted.to(dtype=current_prompt_embeds.dtype)

                        # Add to the first token embedding ([CLS] or BOS)
                        # Shape: [B, 1, H_encoder] + [B, 1, H_encoder]
                        current_prompt_embeds[:, 0:1, :] = current_prompt_embeds[:, 0:1, :] + injection_scale * fidelity_embedding_adjusted.unsqueeze(1)
                        # --------------------------------------------------------

                        prompt_embeds_list.append(current_prompt_embeds)

                # Concatenate embeds from both encoders (output is weight_dtype/fp16)
                encoder_hidden_states = torch.concat(prompt_embeds_list, dim=-1).to(dtype=weight_dtype)
                add_text_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

                # Get original image embeds (latents are float32)
                original_image_embeds = vae.encode(original_pixel_values_vae).latent_dist.sample()

                # Cast original_image_embeds to UNet's expected dtype (fp16) before concat/dropout
                original_image_embeds = original_image_embeds.to(unet.dtype)

                # --- Conditioning Dropout ---
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1).to(encoder_hidden_states.device)

                    # Ensure null_conditioning has correct shape and device
                    current_null_cond = null_conditioning.repeat(bsz // null_conditioning.shape[0] if bsz > null_conditioning.shape[0] else 1, 1, 1)
                    current_null_cond = current_null_cond[:bsz].to(dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device)

                    # Final text conditioning.
                    encoder_hidden_states = torch.where(prompt_mask, current_null_cond, encoder_hidden_states)

                    # Sample masks for the original images
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1).to(original_image_embeds.device)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate noisy_latents (needs casting) and image_embeds (already cast)
                concatenated_noisy_latents = torch.cat([noisy_latents.to(unet.dtype), original_image_embeds], dim=1)

                # Get target (noise is float32)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps) # Target is float32
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Prepare added_cond_kwargs (cast to unet dtype)
                current_add_time_ids = add_time_ids[:bsz].to(device=latents.device, dtype=unet.dtype)
                current_add_text_embeds = add_text_embeds[:bsz].to(device=latents.device, dtype=unet.dtype)
                added_cond_kwargs = {"text_embeds": current_add_text_embeds, "time_ids": current_add_time_ids}

                # Predict the noise residual (UNet operates in fp16)
                model_pred = unet(
                    concatenated_noisy_latents, # fp16
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(unet.dtype), # fp16
                    added_cond_kwargs=added_cond_kwargs, # fp16
                    return_dict=False,
                )[0] # model_pred is fp16

                # Compute loss in float32 for stability
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                 # --- Add Fidelity Loss Component ---
                if args.use_fidelity_loss: # Check if fidelity loss is enabled
                    fidelity_loss_component = torch.tensor(0.0, device=loss.device)
                    # Use fidelity_tensor_for_loss (shape [B]) created earlier

                    if args.no_proxy: # Using Full Fidelity Loss
                        if (global_step + 1) % args.fidelity_loss_freq == 0: # Check frequency
                            with torch.no_grad():
                                # Decode prediction to pixel space
                                pred_latents = noise_scheduler.step(model_pred.detach(), timesteps[0], noisy_latents, return_dict=False)[0]
                                pred_latents = pred_latents.to(dtype=vae.dtype) # Match VAE dtype for decode
                                pred_images = vae.decode(pred_latents / vae.config.scaling_factor, return_dict=False)[0]
                                input_images = batch["original_pixel_values"].to(device=pred_images.device) # Already float32
                                target_images = batch["edited_pixel_values"].to(device=pred_images.device) # Already float32

                                # Initialize LPIPS if needed
                                global lpips_loss_fn
                                if args.fidelity_loss_type == 'lpips' and lpips_loss_fn is None:
                                    logger.info("Initializing LPIPS model for SDXL script...")
                                    lpips_loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)
                                    lpips_loss_fn.eval()

                                # Compute full fidelity loss
                                fidelity_loss_component = compute_fidelity_loss(
                                    pred_images=pred_images,
                                    input_images=input_images,
                                    target_images=target_images,
                                    fidelity_values=fidelity_tensor_for_loss,
                                    loss_type=args.fidelity_loss_type,
                                    lpips_model=lpips_loss_fn if args.fidelity_loss_type == 'lpips' else None,
                                    device=accelerator.device
                                )
                            if accelerator.is_main_process and (global_step + 1) % 100 == 0:
                               logger.info(f"Step {global_step+1}: Using FULL fidelity loss ({args.fidelity_loss_type}): {fidelity_loss_component.item():.4f}")

                    else: # Using Proxy Fidelity Loss
                        if (global_step + 1) % args.fidelity_loss_freq == 0: # Check frequency
                            fidelity_scale = fidelity_tensor_for_loss.view(-1, 1, 1, 1).to(device=model_pred.device)
                            fidelity_loss_component = torch.mean(torch.abs(model_pred.float()) * fidelity_scale.float())
                            2
                            if accelerator.is_main_process and (global_step + 1) % 100 == 0:
                               logger.info(f"Step {global_step+1}: Using PROXY fidelity loss: {fidelity_loss_component.item():.4f}")

                # Scale and add the fidelity loss component
                fidelity_weight = min(args.fidelity_weight, args.fidelity_weight * (global_step / 100))
                loss = loss + fidelity_weight * fidelity_loss_component

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate - This is the crucial step for AMP scaling
                accelerator.backward(loss)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # Combine parameters from both models for a single clipping call
                    params_to_clip = list(unet.parameters()) + list(fidelity_mlp.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    # Step the optimizer (accelerate handles scaler update)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if args.use_ema:
                        ema_unet.step(unet.parameters())

                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0 # Reset loss accumulator

                    # --- Checkpointing ---
                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
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
                            logger.info(f"Saved state to {save_path}")

                    logs = {"step_loss": loss.detach().item(), "lr_unet": lr_scheduler.get_last_lr()[0], "lr_mlp": lr_scheduler.get_last_lr()[-1]}
                    progress_bar.set_postfix(**logs)

                    # --- Validation Trigger ---
                    should_validate = False
                    if accelerator.is_main_process and args.val_images_dir is not None:
                         is_early_step = global_step in [200, 500, 800, 1200]
                         is_regular_step = args.validation_steps > 0 and global_step % args.validation_steps == 0
                         is_final_step = global_step >= args.max_train_steps

                         if is_early_step or is_regular_step or is_final_step:
                             should_validate = True

                    if should_validate:
                        logger.info(f"Triggering validation at step {global_step}")
                        if args.use_ema:
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        # Create pipeline & attach trained fidelity_mlp for validation
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unwrapped_fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)

                        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=unwrapped_unet,
                            text_encoder=text_encoder_1,
                            text_encoder_2=text_encoder_2,
                            tokenizer=tokenizer_1,
                            tokenizer_2=tokenizer_2,
                            vae=vae,
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                            fidelity_mlp=unwrapped_fidelity_mlp,
                        )

                        log_validation(
                            pipeline,
                            args,
                            accelerator,
                            generator,
                            global_step,
                            is_final_validation=(global_step >= args.max_train_steps),
                        )

                        if args.use_ema:
                            ema_unet.restore(unet.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

                if global_step >= args.max_train_steps:
                    break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            vae=vae,
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )

        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if (args.val_image_url_or_path is not None) and (args.validation_prompt is not None):
            log_validation(
                pipeline,
                args,
                accelerator,
                generator,
                global_step,
                is_final_validation=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
