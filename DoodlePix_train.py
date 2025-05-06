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
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from DoodlePix_pipeline import StableDiffusionInstructPix2PixPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from datasets import Dataset, Image
from PIL import Image as PILImage
import glob
from fidelity_mlp import FidelityMLP
from pytorch_msssim import ssim
import lpips
import lmdb
from torch.utils.data import Dataset as TorchDataset
import io
import pickle
import re
import random
if is_wandb_available():
    import wandb

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
                            
                            # with open(os.path.join(step_dir, f"prompt_{base_name}_f{fidelity}.txt"), "w") as f:
                            #     f.write(fidelity_prompt)
                                
                                
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image, resolution):
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
        self.env = None

        logger.info(f"Initializing LMDB dataset from: {lmdb_path}")

        try:
            temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            with temp_env.begin(write=False) as txn:
                self.length = txn.stat()['entries']
                logger.info("Reading keys from LMDB... (This might take a moment)")
                self.keys = [key for key in tqdm(txn.cursor().iternext(keys=True, values=False), total=self.length, desc="Loading LMDB keys")]
            temp_env.close()
            logger.info(f"Found {self.length} entries in LMDB.")
        except lmdb.Error as e:
            logger.error(f"Failed to open or read LMDB at {lmdb_path}: {e}")
            raise IOError(f"Could not initialize LMDB dataset at {lmdb_path}")

    def _init_db(self):

        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()

        key = self.keys[index]
        with self.env.begin(write=False) as txn:
            serialized_data = txn.get(key)

        if serialized_data is None:
             raise KeyError(f"Key {key.decode('utf-8')} not found in LMDB (index {index}). This shouldn't happen.")

        try:
            input_img_bytes, edited_img_bytes, edit_prompt = pickle.loads(serialized_data)
        except pickle.UnpicklingError as e:
            raise RuntimeError(f"Failed to unpickle data for key {key.decode('utf-8')}: {e}")

        try:
            input_image = PILImage.open(io.BytesIO(input_img_bytes)).convert('RGB')
            edited_image = PILImage.open(io.BytesIO(edited_img_bytes)).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to decode image for key {key.decode('utf-8')}: {e}")

        sample = {
            'input_image': input_image,
            'edited_image': edited_image,
            'edit_prompt': edit_prompt
        }

        if self.transform:
             sample = self.transform(sample)

        return sample

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
    
    def tokenize_captions(captions):
        """Helper function to tokenize captions using the loaded tokenizer."""
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    
    if args.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    

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

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

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
                model = models.pop()

                if isinstance(model, FidelityMLP):
                    fidelity_mlp_path = os.path.join(input_dir, "fidelity_mlp")
                    if os.path.exists(fidelity_mlp_path):
                        load_model = FidelityMLP.from_pretrained(fidelity_mlp_path)
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                else:
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

    if args.fidelity_mlp_path:
        if os.path.exists(args.fidelity_mlp_path):
            logger.info(f"Loading FidelityMLP from {args.fidelity_mlp_path}")
            fidelity_mlp = FidelityMLP.from_pretrained(args.fidelity_mlp_path)
        else:
            logger.info(f"Creating new FidelityMLP and will save to {args.fidelity_mlp_path}")
            hidden_size = text_encoder.config.hidden_size
            fidelity_mlp = FidelityMLP(hidden_size)
            os.makedirs(args.fidelity_mlp_path, exist_ok=True)
    else:
        logger.info("Creating new FidelityMLP")
        hidden_size = text_encoder.config.hidden_size
        fidelity_mlp = FidelityMLP(hidden_size)

    logger.info("This has Normalized Range 0-9")
    
    fidelity_mlp.to(accelerator.device)


    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
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
    

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            # random rotation of +90 or -90 degrees for 10% of images
            transforms.Lambda(lambda x: transforms.functional.rotate(x, 90 * (2 * torch.randint(0, 2, (1,)).item() - 1)) 
                             if torch.rand(1).item() < 0.1 else x),
        ]
    )
    if args.train_data_dir is None:
        raise ValueError("--train_data_dir must be specified to point to the LMDB database.")
    if args.dataset_name is not None:
        logger.warning("--dataset_name is ignored when --train_data_dir is provided for LMDB loading.")


    def lmdb_transform(sample):
        """
        Applies transformations to a single sample dict loaded from LMDB.
        Ensures consistent random transformations for image pairs and converts to tensor.
        """
        input_pil = sample['input_image']
        edited_pil = sample['edited_image']
        prompt = sample['edit_prompt']

        image_tensor_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        )

        seed = random.randint(0, 2**32 - 1)

        torch.manual_seed(seed)
        random.seed(seed)
        input_pil_augmented = train_transforms(input_pil)

        torch.manual_seed(seed)
        random.seed(seed)
        edited_pil_augmented = train_transforms(edited_pil)

        original_pixel_values = image_tensor_transforms(input_pil_augmented)
        edited_pixel_values = image_tensor_transforms(edited_pil_augmented)

        input_ids = tokenize_captions([prompt])[0]

        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    logger.info(f"Loading dataset using LMDBImagePromptDataset from: {args.train_data_dir}")
    try:
        train_dataset = LMDBImagePromptDataset(lmdb_path=args.train_data_dir, transform=lmdb_transform)
    except Exception as e:
        logger.error(f"Failed to initialize LMDB dataset at {args.train_data_dir}: {e}", exc_info=True)
        raise

    if args.max_train_samples is not None:
        if args.max_train_samples < len(train_dataset):
            logger.info(f"Truncating LMDB dataset to {args.max_train_samples} samples.")
            from torch.utils.data import Subset

            train_dataset = Subset(train_dataset, list(range(args.max_train_samples)))
        else:
             logger.info(f"max_train_samples ({args.max_train_samples}) >= dataset size ({len(train_dataset)}), using full dataset.")

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

    unet, optimizer, train_dataloader, lr_scheduler, fidelity_mlp = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, fidelity_mlp
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=torch.float32)
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
                
                fidelity_values = []
                prompts_text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                for prompt_text in prompts_text:
                    match = re.search(r"f\s*=?\s*(\d+)|f(\d+)", prompt_text, re.IGNORECASE)
                    if match:
                        f_int = int(match.group(1) if match.group(1) else match.group(2))
                        f_int = max(0, min(f_int, 9)) 
                        fidelity_val = f_int / 9.0
                    else:
                        logger.warning(f"Could not parse f-value from prompt: '{prompt_text}'. Defaulting to f5 (normalized 0.5).")
                        f_int = 5
                        fidelity_val = 5.0 / 9.0
                    fidelity_values.append(fidelity_val)

                fidelity_tensor = torch.tensor(fidelity_values, device=latents.device, dtype=latents.dtype).unsqueeze(-1) # Shape [B, 1]
                try:
                    fidelity_mlp_module = accelerator.unwrap_model(fidelity_mlp)
                except:
                    fidelity_mlp_module = fidelity_mlp
                    print("the fidelity_mlp_module is", fidelity_mlp_module)
                fidelity_embedding = fidelity_mlp_module(fidelity_tensor)
                injection_scale = args.fidelity_weight
                encoder_hidden_states[:, 0:1, :] = encoder_hidden_states[:, 0:1, :] + injection_scale * fidelity_embedding.unsqueeze(1)
                
                fidelity_tensor = fidelity_tensor.squeeze(-1)
                
#--------CONDITIONING DROPOUT TO SUPPORT CLASSIFIER-FREE GUIDANCE DURING INFERENCE.-------#
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
                if accelerator.is_main_process and global_step % 100 == 0:
                    logger.info(f"Step {global_step}: Fidelity embedding stats: mean={fidelity_embedding.mean().item():.4f}, std={fidelity_embedding.std().item():.4f}")
            
                
                fidelity_loss = torch.tensor(0.0, device=loss.device)

#---------COMPUTE FIDELITY LOSS---------#
                if args.no_proxy:
                    if global_step % 10 == 0:
                        with torch.no_grad():
                            pred_latents = noise_scheduler.step(
                                model_pred.detach(), timesteps[0], noisy_latents, return_dict=False
                            )[0]
                            
                            pred_latents = pred_latents.to(dtype=weight_dtype)
                            
                            pred_images = vae.decode(pred_latents / vae.config.scaling_factor, return_dict=False)[0]
                            
                            input_images = batch["original_pixel_values"].to(device=pred_images.device, dtype=pred_images.dtype)
                            target_images = batch["edited_pixel_values"].to(device=pred_images.device, dtype=pred_images.dtype)
                            
                            global lpips_loss_fn
                            if args.fidelity_loss_type == 'lpips' and lpips_loss_fn is None:
                                logger.info("Initializing LPIPS model...")
                                lpips_loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)
                                lpips_loss_fn.eval()
                            
                            fidelity_loss = compute_fidelity_loss(
                                pred_images=pred_images,
                                input_images=input_images,
                                target_images=target_images,
                                fidelity_values=fidelity_tensor,
                                loss_type=args.fidelity_loss_type,
                                lpips_model=lpips_loss_fn if args.fidelity_loss_type == 'lpips' else None,
                                device=accelerator.device
                            )
                            
                            if accelerator.is_main_process and global_step % 100 == 0:
                                logger.info(f"Step {global_step}: Sample extracted fidelity values: {fidelity_values[:4]}")
                                logger.info(f"Step {global_step}: Fidelity tensor for loss: {fidelity_tensor[:4]}")
                else:
                    fidelity_scale = fidelity_tensor.view(-1, 1, 1, 1)
                    fidelity_loss = torch.mean(torch.abs(model_pred) * fidelity_scale)
                    if accelerator.is_main_process and global_step % 100 == 0:
                        logger.info(f"Step {global_step}: Using PROXY fidelity loss: {fidelity_loss.item():.4f}")

                fidelity_weight = min(args.fidelity_weight, args.fidelity_weight * (global_step / 200))
                
                if accelerator.is_main_process and global_step % 100 == 0:
                    main_loss_val = F.mse_loss(model_pred.float(), target.float(), reduction="mean").item()
                    fidelity_comp_val = (fidelity_loss * fidelity_weight).item()
                    logger.info(f"Step {global_step}: Main Loss: {main_loss_val:.6f}, Weighted Fidelity Comp: {fidelity_comp_val:.6f}")

                loss = loss + fidelity_weight * fidelity_loss

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                if accelerator.is_main_process:
                    if args.checkpointing_steps is not None and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

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

#---------VALIDATION---------#
                if accelerator.is_main_process:
                    if args.val_images_dir is not None and (
                        (global_step == 100) or
                        (global_step == 600) or
                        (global_step == 1500) or
                        (args.validation_steps > 0 and global_step % args.validation_steps == 0) or
                        (global_step >= args.max_train_steps)
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
                        pipeline.fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)
                        
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
                (global_step == 100) or
                (args.validation_steps > 0 and global_step % args.validation_steps == 0) or
                (global_step >= args.max_train_steps)
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
                pipeline.fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)

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

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        unet = unwrap_model(unet)
        text_encoder = unwrap_model(text_encoder)

        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)
        pipeline.save_pretrained(args.output_dir)
        
        accelerator.unwrap_model(fidelity_mlp).save_pretrained(
            os.path.join(args.output_dir, "fidelity_mlp")
        )

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

        if args.fidelity_mlp_path:
            if accelerator.is_main_process:
                unwrapped_fidelity_mlp = accelerator.unwrap_model(fidelity_mlp)
                unwrapped_fidelity_mlp.save_pretrained(args.fidelity_mlp_path)
                logger.info(f"Saved FidelityMLP to {args.fidelity_mlp_path}")

    if global_step % args.validation_steps == 0:
        test_fidelities = [0.1, 0.5, 0.9]
        with torch.no_grad():
            for f_val in test_fidelities:
                test_tensor = torch.tensor([[f_val]], device=accelerator.device)
                output = fidelity_mlp(test_tensor)
                logger.info(f"Fidelity {f_val} produces embedding with mean: {output.mean().item()}, std: {output.std().item()}")

    accelerator.end_training()


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


if __name__ == "__main__":
    main()
