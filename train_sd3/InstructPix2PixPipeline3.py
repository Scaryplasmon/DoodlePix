"""
    Custom implementation of InstructPix2Pix pipeline script from the diffusers repo
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py

"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast, CLIPTextModelWithProjection, SiglipImageProcessor, SiglipVisionModel
import os


from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ImageProjection, SD3Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import PIL_INTERPOLATION, deprecate, is_torch_xla_available, logging
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from . import StableDiffusionPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class StableDiffusion3InstructPix2PixPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    StableDiffusionLoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for pixel-level image editing by following text instructions (based on Stable Diffusion).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`SD3Transformer2DModel`]):
            A `SD3Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) for
            more details about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "image_latents"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        text_encoder_2: CLIPTextModelWithProjection,
        text_encoder_3: T5EncoderModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        tokenizer_3: T5TokenizerFast,
        image_encoder: Optional[SiglipVisionModel] = None,
        feature_extractor: Optional[SiglipImageProcessor] = None,
        fidelity_mlp = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.fidelity_mlp = fidelity_mlp
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 77

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
                image latents as `image`, but if passing latents directly it is not encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Push the generated image towards the initial `image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

## FORWARD PASS

        # 0. Check inputs
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        device = self._execution_device

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
##NOT NEEDED FOR SD3
        # # 7. Check that shapes of latents and image match the UNet channels
        # num_channels_image = image_latents.shape[1]
        # if num_channels_latents + num_channels_image != self.unet.config.in_channels:
        #     raise ValueError(
        #         f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
        #         f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
        #         f" `num_channels_image`: {num_channels_image} "
        #         f" = {num_channels_latents+num_channels_image}. Please verify the config of"
        #         " `pipeline.unet` or your `image` input."
        #     )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8.1 Add image embeds for IP-Adapter
        added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = torch.cat([latents] * 3) if self.do_classifier_free_guidance else latents

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    hidden_states=scaled_latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + self.guidance_scale * (noise_pred_text - noise_pred_image)
                        + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    image_latents = callback_outputs.pop("image_latents", image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        prompt_2 = prompt
        prompt_3 = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # Normalize prompts
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3
            
            # Get CLIP1 embeddings
            prompt_embed_1, pooled_prompt_embed_1 = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            
            # Get CLIP2 embeddings
            prompt_embed_2, pooled_prompt_embed_2 = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            
            # Combine CLIP embeddings
            clip_prompt_embeds = torch.cat([prompt_embed_1, prompt_embed_2], dim=-1)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed_1, pooled_prompt_embed_2], dim=-1)

            # Get T5 embeddings
            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=256,
                device=device,
            )

            # Pad CLIP embeddings to match T5 dimensions
            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            # Concatenate along sequence dimension
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=1)
            
            # Apply fidelity using the more sophisticated helper method
            # This handles different embedding dimensions properly
        if self.fidelity_mlp is not None:
            # First handle CLIP embeddings (first token)
            prompt_embeds = self._inject_fidelity_into_first_token(
                prompt_embeds=prompt_embeds,
                prompt=prompt,
                device=device,
                scale=0.2  # Same scale as original
            )
            
            # Then handle T5 embeddings if needed - get the starting index
            clip_seq_len = clip_prompt_embeds.shape[1]  # This is the CLIP sequence length
            prompt_embeds = self._inject_fidelity_into_t5(
                prompt_embeds=prompt_embeds,
                prompt=prompt,
                device=device,
                t5_start_index=clip_seq_len,  # T5 starts after CLIP
                scale=0.2  # Same scale as original
            )
        
        # Handle classifier-free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt
            negative_prompt_3 = negative_prompt

            # Normalize to lists
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            negative_prompt_3 = batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3

            # Get negative embeddings from first CLIP
            negative_prompt_embed_1, negative_pooled_prompt_embed_1 = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            
            # Get negative embeddings from second CLIP
            negative_prompt_embed_2, negative_pooled_prompt_embed_2 = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            
            # Combine CLIP negative embeddings
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed_1, negative_prompt_embed_2], dim=-1)

            # Get T5 negative embeddings
            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=256,
                device=device,
            )

            # Pad negative CLIP embeddings
            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            # Combine all negative embeddings
            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=1)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed_1, negative_pooled_prompt_embed_2], dim=-1
            )

            # Apply the fidelity to negative embeddings in the same way
            if self.fidelity_mlp is not None:
                # First token fidelity
                negative_prompt_embeds = self._inject_fidelity_into_first_token(
                    prompt_embeds=negative_prompt_embeds,
                    prompt=negative_prompt,
                    device=device,
                    scale=0.2
                )
                
                # T5 fidelity at the correct position
                clip_seq_len = negative_clip_prompt_embeds.shape[1]
                negative_prompt_embeds = self._inject_fidelity_into_t5(
                    prompt_embeds=negative_prompt_embeds,
                    prompt=negative_prompt,
                    device=device,
                    t5_start_index=clip_seq_len,
                    scale=0.2
                )

            # Combine for classifier-free guidance
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

        return prompt_embeds, pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat(
                        [single_image_embeds, single_negative_image_embeds, single_negative_image_embeds]
                    )
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    (
                        single_image_embeds,
                        single_negative_image_embeds,
                        single_negative_image_embeds,
                    ) = single_image_embeds.chunk(3)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat(
                        [single_image_embeds, single_negative_image_embeds, single_negative_image_embeds]
                    )
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1.0 and self.image_guidance_scale >= 1.0

    @classmethod
    def from_pretrained_with_fidelity(cls, pretrained_model_path, fidelity_mlp_path=None, **kwargs):
        """Load the pipeline with a fidelity MLP model."""
        pipeline = cls.from_pretrained(pretrained_model_path, **kwargs)
        
        if fidelity_mlp_path is not None:
            from fidelity_mlp import FidelityMLP
            # Load the FidelityMLP
            hidden_size = pipeline.text_encoder.config.hidden_size
            if os.path.exists(fidelity_mlp_path):
                pipeline.fidelity_mlp = FidelityMLP.from_pretrained(fidelity_mlp_path)
            else:
                # Create a new one if path doesn't exist
                pipeline.fidelity_mlp = FidelityMLP(hidden_size)
                logger.warning(
                    f"Fidelity MLP not found at {fidelity_mlp_path}, initialized a new one with hidden size {hidden_size}"
                )
            
            # Move to the same device as the pipeline
            pipeline.fidelity_mlp.to(pipeline.device)
        
        return pipeline

    ###############################################################################
    # 1) HELPER FUNCTIONS FOR CLIP + T5 ENCODING
    ###############################################################################
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds


    def _merge_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int,
        clip_skip: Optional[int] = None,
        max_length_t5: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1) Encode prompt with CLIP1, CLIP2 -> cat them in last dimension
        2) Encode prompt with T5
        3) Pad whichever is narrower so that both sets have same hidden_dim
        4) cat them across seq_len dimension
        5) Return final_prompt_embeds and final_pooled_embeds
        """
        # step 1: get CLIP1 + CLIP2
        clip_embeds_1, pooled_1 = self._get_clip_prompt_embeds(
            prompt=prompt, 
            tokenizer_idx=0,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            device=device
        )
        clip_embeds_2, pooled_2 = self._get_clip_prompt_embeds(
            prompt=prompt, 
            tokenizer_idx=1,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            device=device
        )
        # combine them on the last dimension
        # shape e.g.: [batch * num_images, seq_len, (clipdim1+clipdim2)]
        clip_prompt_embeds = torch.cat([clip_embeds_1, clip_embeds_2], dim=-1)
        clip_pooled_embeds = torch.cat([pooled_1, pooled_2], dim=-1)

        # step 2: get T5
        t5_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            max_length=max_length_t5,
        )
        # shape e.g.: [batch*num_images, t5_seq_len, t5_dim]

        # step 3: if their last-dims differ, we can pad one side 
        clip_dim = clip_prompt_embeds.shape[-1]
        t5_dim   = t5_embeds.shape[-1]
        if clip_dim < t5_dim:
            # pad clip in the last dimension so it becomes [..., t5_dim]
            pad_amt = t5_dim - clip_dim
            clip_prompt_embeds = F.pad(clip_prompt_embeds, (0, pad_amt, 0, 0, 0, 0), value=0.0)
            clip_pooled_embeds = F.pad(clip_pooled_embeds,  (0, pad_amt), value=0.0)
        elif t5_dim < clip_dim:
            # pad t5
            pad_amt = clip_dim - t5_dim
            t5_embeds = F.pad(t5_embeds, (0, pad_amt, 0, 0, 0, 0), value=0.0)

        # step 4: cat across seq_len
        # shape e.g.: [batch*num_images, clip_seq_len + t5_seq_len, max(clip_dim, t5_dim)]
        final_prompt_embeds = torch.cat([clip_prompt_embeds, t5_embeds], dim=1)

        # step 5: return
        return final_prompt_embeds, clip_pooled_embeds
    ###############################################################################
    # 2) HELPER FUNCTIONS FOR FIDELITY MLP
    ###############################################################################
    def _inject_fidelity_into_first_token(
        self,
        prompt_embeds: torch.Tensor,
        prompt: Union[str, List[str]],
        device: torch.device,
        scale: float = 0.2,
    ) -> torch.Tensor:
        """
        Finds the numeric 'f#' in `prompt`, uses `self.fidelity_mlp` to get an embedding,
        then adds it to the first token of the entire 'prompt_embeds'.
        
        This modifies `prompt_embeds` in-place.
        """
        if self.fidelity_mlp is None:
            return prompt_embeds  # no-op if we don't have a fidelity_mlp

        # 1) find numeric fidelity value from prompt
        fidelity_val = 0.5
        if prompt is not None:
            import re
            if isinstance(prompt, str):
                match = re.search(r"f\s*=?\s*(\d+)|f(\d+)", prompt, re.IGNORECASE)
                if match:
                    f_int = int(match.group(1) or match.group(2))
                    f_int = max(1, min(f_int, 9))
                    fidelity_val = 0.1 + (f_int - 1) * (0.8 / 8)
            elif isinstance(prompt, list):
                f_vals = []
                for p in prompt:
                    match = re.search(r"f\s*=?\s*(\d+)|f(\d+)", p, re.IGNORECASE)
                    if match:
                        f_int = int(match.group(1) or match.group(2))
                        f_int = max(1, min(f_int, 9))
                        f_vals.append(0.1 + (f_int - 1) * (0.8 / 8))
                if f_vals:
                    fidelity_val = sum(f_vals) / len(f_vals)

        # 2) Create fidelity tensor & run the MLP, passing the target dimension
        batch_size = prompt_embeds.shape[0]
        token_dim = prompt_embeds.shape[-1]  # Get the actual embedding dimension we need to match
        
        fidelity_tensor = torch.full((batch_size, 1), fidelity_val, device=device, dtype=prompt_embeds.dtype)
        # Pass the target dimension to get correctly sized output
        fidelity_embedding = self.fidelity_mlp(fidelity_tensor, target_dim=token_dim)

        # 3) inject into the first token
        prompt_embeds[:, 0] += scale * fidelity_embedding

        return prompt_embeds

    def _inject_fidelity_into_t5(
        self,
        prompt_embeds: torch.Tensor,
        prompt: Union[str, List[str]],
        device: torch.device,
        t5_start_index: int,
        scale: float = 0.2,
    ) -> torch.Tensor:
        """
        Similar to the first helper, but only modifies the T5 portion's first token.
        We assume the T5 portion starts at index t5_start_index along sequence dimension.
        E.g. if we did final_embeds = cat([clip_embeds, t5_embeds], dim=1),
        then t5_start_index = clip_embeds.shape[1].
        """

        if self.fidelity_mlp is None:
            return prompt_embeds

        # 1) parse fidelity from prompt
        fidelity_val = 0.5
        if prompt is not None:
            import re
            if isinstance(prompt, str):
                match = re.search(r"f\s*=?\s*(\d+)|f(\d+)", prompt, re.IGNORECASE)
                if match:
                    f_int = int(match.group(1) or match.group(2))
                    f_int = max(1, min(f_int, 9))
                    fidelity_val = 0.1 + (f_int - 1) * (0.8 / 8)
            elif isinstance(prompt, list):
                f_vals = []
                for p in prompt:
                    match = re.search(r"f\s*=?\s*(\d+)|f(\d+)", p, re.IGNORECASE)
                    if match:
                        f_int = int(match.group(1) or match.group(2))
                        f_int = max(1, min(f_int, 9))
                        f_vals.append(0.1 + (f_int - 1) * (0.8 / 8))
                if f_vals:
                    fidelity_val = sum(f_vals) / len(f_vals)

        # 2) run MLP
        batch_size = prompt_embeds.shape[0]
        token_dim = prompt_embeds.shape[-1]  # Get the actual embedding dimension we need to match
        fidelity_tensor = torch.full((batch_size, 1), fidelity_val, device=device, dtype=prompt_embeds.dtype)
        fidelity_embedding = self.fidelity_mlp(fidelity_tensor, target_dim=token_dim)

        # 3) add to T5's first token
        prompt_embeds[:, t5_start_index] += scale * fidelity_embedding
        return prompt_embeds
