from PIL import Image, ImageOps
import numpy as np
from types import SimpleNamespace
import os
import torch
from fidelity_mlp import FidelityMLP

class InferenceHandler:
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.schedulers = None
        self._setup_done = False
        self.base_model_path = None
        self.lora_path = None
        self.is_sdxl = False # Flag to track model type
        self._torch = None
        self._sd_pipeline_class = None
        self._sdxl_pipeline_class = None
        self._euler_ancestral_scheduler = None

    def _lazy_setup(self):
        """Lazy load ML dependencies only when needed"""
        if self._setup_done:
            return

        import torch
        # Import both pipeline types and the scheduler
        from diffusers import (
            StableDiffusionInstructPix2PixPipeline,
            EulerAncestralDiscreteScheduler,
        )
        try:
            from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_instruct_pix2pix import (
                StableDiffusionXLInstructPix2PixPipeline,
            )
            print("Using custom SDXL InstructPix2Pix pipeline.")
        except ImportError:
            print("Warning: Custom SDXL pipeline not found. Falling back to diffusers version.")
            self._sdxl_pipeline_class = StableDiffusionXLInstructPix2PixPipeline


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.schedulers = {
            "Euler Ancestral": EulerAncestralDiscreteScheduler,
        }

        self._torch = torch
        self._sd_pipeline_class = StableDiffusionInstructPix2PixPipeline
        self._sdxl_pipeline_class = StableDiffusionXLInstructPix2PixPipeline
        self._euler_ancestral_scheduler = EulerAncestralDiscreteScheduler
        self._setup_done = True

    def get_scheduler_names(self):
        """Get list of available schedulers without loading ML stuff"""
        # For now, assume Euler Ancestral is compatible with both
        return ["Euler Ancestral"]

    def load_model(self, model_path, scheduler_name="Euler Ancestral", is_sdxl=False):
        """Load the base model from path, specifying if it's SDXL"""
        try:
            self._lazy_setup()

            self.is_sdxl = is_sdxl # Store the model type
            pipeline_class = self._sdxl_pipeline_class if is_sdxl else self._sd_pipeline_class

            scheduler_class = self.schedulers.get(scheduler_name)
            if not scheduler_class:
                 print(f"Warning: Scheduler '{scheduler_name}' not found. Using Euler Ancestral.")
                 scheduler_class = self._euler_ancestral_scheduler

            # Check if scheduler exists at the path, otherwise load default config
            scheduler_path = os.path.join(model_path, "scheduler")
            if os.path.isdir(scheduler_path):
                 scheduler = scheduler_class.from_pretrained(model_path, subfolder="scheduler")
            else:
                 print(f"Scheduler config not found in {model_path}. Using default config for {scheduler_name}.")
                 # This might need adjustment depending on scheduler type
                 scheduler = scheduler_class.from_config(pipeline_class.load_config(model_path, subfolder="scheduler"))


            fidelity_mlp_path = os.path.join(model_path, "fidelity_mlp")
            fidelity_mlp_instance = None

            # Check and load fidelity MLP first if it exists
            if os.path.exists(fidelity_mlp_path):
                print(f"Found potential fidelity MLP at: {fidelity_mlp_path}")
                try:
                    fidelity_mlp_instance = FidelityMLP.from_pretrained(fidelity_mlp_path)
                    fidelity_mlp_instance = fidelity_mlp_instance.to(device=self.device, dtype=self._torch.float16)
                    print(f"Loaded fidelity MLP from: {fidelity_mlp_path}")
                except Exception as mlp_e:
                    print(f"Warning: Failed to load fidelity MLP from {fidelity_mlp_path}: {mlp_e}")
                    fidelity_mlp_instance = None


            # Load the appropriate pipeline
            if is_sdxl and hasattr(pipeline_class, 'from_pretrained_with_fidelity') and fidelity_mlp_instance:
                 print("Loading SDXL pipeline using from_pretrained_with_fidelity...")
                 # If using the custom pipeline with the special loader method:
                 self.pipeline = pipeline_class.from_pretrained_with_fidelity(
                     model_path,
                     fidelity_mlp_path=fidelity_mlp_path, # Pass the path again for the method
                     torch_dtype=self._torch.float16,
                     safety_checker=None,
                     scheduler=scheduler
                 ).to(self.device)
            elif is_sdxl and fidelity_mlp_instance:
                 print("Loading SDXL pipeline and manually attaching MLP...")
                 # Load standard SDXL pipeline first
                 self.pipeline = pipeline_class.from_pretrained(
                     model_path,
                     torch_dtype=self._torch.float16,
                     safety_checker=None,
                     scheduler=scheduler
                 )
                 # Manually attach the loaded MLP
                 self.pipeline.fidelity_mlp = fidelity_mlp_instance
                 self.pipeline.to(self.device)
                 print("Manually attached fidelity MLP to SDXL pipeline.")
            else:
                 # Load standard SD 1.5 or SDXL without MLP
                 print(f"Loading {'SDXL' if is_sdxl else 'SD1.5'} pipeline...")
                 self.pipeline = pipeline_class.from_pretrained(
                     model_path,
                     torch_dtype=self._torch.float16,
                     safety_checker=None,
                     scheduler=scheduler
                 )
                 # If it's SD1.5 and we found an MLP, attach it manually
                 if not is_sdxl and fidelity_mlp_instance:
                     self.pipeline.fidelity_mlp = fidelity_mlp_instance
                     print("Manually attached fidelity MLP to SD1.5 pipeline.")
                 elif not fidelity_mlp_instance:
                      print("No Fidelity MLP found or loaded.")

                 self.pipeline.to(self.device)


            # Store base model path
            self.base_model_path = model_path
            self.lora_path = None # Reset LoRA path when loading a new base model

            # Enable optimizations
            # self.pipeline.enable_model_cpu_offload() # Recommended for lower VRAM
            # self.pipeline.enable_attention_slicing() # Fallback if xformers not available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                 try:
                     self.pipeline.enable_xformers_memory_efficient_attention()
                     print("Enabled xformers memory efficient attention.")
                 except Exception as xformers_e:
                     print(f"Could not enable xformers: {xformers_e}. Using attention slicing.")

            print(f"Model {'SDXL' if is_sdxl else 'SD1.5'} loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            # Clean up potentially partially loaded state
            self.pipeline = None
            self.base_model_path = None
            self.is_sdxl = False
            return False


    def load_lora(self, lora_path, intensity=1.0):
        """
        Load LoRA weights onto the current model with specified intensity
        
        Args:
            lora_path: Path to LoRA weights
            intensity: Float between 0.0-2.0 controlling strength of LoRA effect (default: 1.0)
        """
        if not self.pipeline:
            return False, "Please load a base model first"
            
        try:
            # First unload any existing LoRA weights to avoid conflicts
            # Keep the safer unload logic from the previous attempt
            try:
                 if self.lora_path: # Check if a path exists before attempting unload
                     print(f"Attempting to unload previous LoRA: {self.lora_path}")
                     self.pipeline.unload_lora_weights()
                     print(f"Unloaded previous LoRA.")
                 # Reset kwargs regardless of whether unload worked or was needed
                 if hasattr(self.pipeline, 'cross_attention_kwargs'):
                      self.pipeline.cross_attention_kwargs = {}
            except Exception as unload_e:
                 print(f"Ignoring error during unload attempt: {unload_e}. Resetting LoRA path and kwargs.")
                 # Ensure lora_path and kwargs are reset even if unload fails
                 self.lora_path = None
                 if hasattr(self.pipeline, 'cross_attention_kwargs'):
                     self.pipeline.cross_attention_kwargs = {}

            
            print(f"Loading LoRA from: {lora_path} with intensity {intensity}")
            # Load LoRA weights with cross_attention_kwargs to set scale immediately
            # This is the key part from your original script
            self.pipeline.load_lora_weights(
                lora_path,
                # The following line applies the scale during load for some diffusers versions/methods
                # For safety, we'll also explicitly set it after load in set_lora_intensity
                cross_attention_kwargs={"scale": intensity}
            )
            
            self.lora_path = lora_path

            # Check if there's a fidelity MLP in the LoRA folder
            fidelity_mlp_path = os.path.join(lora_path, "fidelity_mlp")
            if os.path.exists(fidelity_mlp_path):
                 try: # Added try-except for robustness
                    self.pipeline.fidelity_mlp = FidelityMLP.from_pretrained(fidelity_mlp_path)
                    self.pipeline.fidelity_mlp = self.pipeline.fidelity_mlp.to(device=self.device, dtype=self._torch.float16)
                    print(f"Loaded fidelity MLP from LoRA: {fidelity_mlp_path}")
                 except Exception as lora_mlp_e:
                    print(f"Warning: Failed to load/attach LoRA MLP {fidelity_mlp_path}: {lora_mlp_e}")
                    # If LoRA MLP fails to load, try reloading base MLP if available
                    base_mlp_path = os.path.join(self.base_model_path, "fidelity_mlp")
                    if os.path.exists(base_mlp_path):
                        try:
                            print("Re-applying base model's fidelity MLP as LoRA MLP failed.")
                            base_mlp = FidelityMLP.from_pretrained(base_mlp_path)
                            self.pipeline.fidelity_mlp = base_mlp.to(device=self.device, dtype=self._torch.float16)
                        except Exception as base_mlp_reload_e:
                             print(f"Warning: Failed to reload base fidelity MLP: {base_mlp_reload_e}")
                             if hasattr(self.pipeline, 'fidelity_mlp'): del self.pipeline.fidelity_mlp # Ensure no broken MLP attached
                    else:
                         if hasattr(self.pipeline, 'fidelity_mlp'): del self.pipeline.fidelity_mlp # Ensure no broken MLP attached
            else:
                 # If LoRA doesn't have an MLP, reload base MLP if available
                 base_mlp_path = os.path.join(self.base_model_path, "fidelity_mlp")
                 if os.path.exists(base_mlp_path):
                     try:
                         print("Re-applying base model's fidelity MLP as LoRA has none.")
                         base_mlp = FidelityMLP.from_pretrained(base_mlp_path)
                         self.pipeline.fidelity_mlp = base_mlp.to(device=self.device, dtype=self._torch.float16)
                     except Exception as base_mlp_reload_e:
                         print(f"Warning: Failed to reload base fidelity MLP: {base_mlp_reload_e}")
                         if hasattr(self.pipeline, 'fidelity_mlp'): del self.pipeline.fidelity_mlp
                 else:
                     # Ensure no previous MLP lingers if neither LoRA nor base has one
                     if hasattr(self.pipeline, 'fidelity_mlp'):
                          print("Removing previous fidelity MLP as neither LoRA nor base model has one.")
                          del self.pipeline.fidelity_mlp


            # Explicitly set the cross_attention_kwargs after loading
            self.pipeline.cross_attention_kwargs = {"scale": intensity}
            
            return True, f"LoRA weights loaded from {lora_path} with intensity {intensity}"
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            import traceback
            traceback.print_exc()
             # Attempt to revert state
            self.lora_path = None
            if hasattr(self.pipeline, 'cross_attention_kwargs'):
                self.pipeline.cross_attention_kwargs = {} # Reset kwargs
            try: # Try unloading again if load failed mid-way
                self.pipeline.unload_lora_weights()
            except: pass
            return False, f"Error loading LoRA: {str(e)}"

    def set_lora_intensity(self, intensity):
        """
        Adjust the intensity/scale of loaded LoRA weights
        
        Args:
            intensity: Float between 0.0-2.0 controlling strength of LoRA effect
        
        Returns:
            bool: Success status
        """
        if not self.pipeline or not self.lora_path:
            return False, "No LoRA weights loaded"
            
        try:
            # Set the scale through the pipeline's cross_attention_kwargs
            # This is the core logic from your original script
            self.pipeline.cross_attention_kwargs = {"scale": intensity}
            print(f"Set LoRA intensity using cross_attention_kwargs to {intensity}")
            return True, f"LoRA intensity set to {intensity}"
        except Exception as e:
            print(f"Error setting LoRA intensity: {e}")
            return False, f"Error setting LoRA intensity: {str(e)}"

    def unload_lora(self):
        """Unload LoRA weights and return to base model"""
        if not self.pipeline or not self.lora_path:
            print("No pipeline loaded or no LoRA path recorded to unload.")
            # Return True if no LoRA is loaded, as the state is already "unloaded"
            return True if not self.lora_path else False 

        try:
            print(f"Unloading LoRA weights: {self.lora_path}")
            # Use proper API to unload LoRA
            self.pipeline.unload_lora_weights()
            self.lora_path = None
             # Clear cross_attention_kwargs scale after unloading
            if hasattr(self.pipeline, 'cross_attention_kwargs'):
                 self.pipeline.cross_attention_kwargs = {} # Reset kwargs
            print("LoRA unloaded successfully.")

            # Reload base model's fidelity MLP if it exists
            base_mlp_path = os.path.join(self.base_model_path, "fidelity_mlp")
            if os.path.exists(base_mlp_path):
                 try:
                     print("Reloading base model's fidelity MLP after unloading LoRA.")
                     base_mlp = FidelityMLP.from_pretrained(base_mlp_path)
                     self.pipeline.fidelity_mlp = base_mlp.to(device=self.device, dtype=self._torch.float16)
                 except Exception as base_mlp_reload_e:
                     print(f"Warning: Failed to reload base fidelity MLP: {base_mlp_reload_e}")
                     # Ensure MLP attribute doesn't linger if reload fails but existed before
                     if hasattr(self.pipeline, 'fidelity_mlp'):
                          delattr(self.pipeline, 'fidelity_mlp')
            else:
                 # Ensure no MLP attribute lingers if base model doesn't have one
                 if hasattr(self.pipeline, 'fidelity_mlp'):
                      print("Removing fidelity MLP as base model does not have one.")
                      delattr(self.pipeline, 'fidelity_mlp')


            return True
        except Exception as e:
            print(f"Error unloading LoRA: {e}")
            # State might be uncertain here, but clear lora_path anyway
            self.lora_path = None
            if hasattr(self.pipeline, 'cross_attention_kwargs'):
                self.pipeline.cross_attention_kwargs = {}
            return False

    def change_scheduler(self, scheduler_name):
        """Change scheduler if pipeline is loaded"""
        if self.pipeline and self._setup_done:
             scheduler_class = self.schedulers.get(scheduler_name)
             if scheduler_class:
                 print(f"Changing scheduler to {scheduler_name}")
                 self.pipeline.scheduler = scheduler_class.from_config(
                     self.pipeline.scheduler.config
                 )
                 return True
             else:
                 print(f"Scheduler {scheduler_name} not recognized.")
                 return False
        else:
             print("Pipeline not loaded, cannot change scheduler.")
             return False

    def generate_image(self, drawing, props, callback=None):
        """Generate image from drawing"""
        if not self.pipeline:
            raise ValueError("Pipeline not loaded! Please load a model first.")
        self._lazy_setup() # Ensure torch is loaded

        # --- Image Preparation ---
        if not isinstance(drawing, Image.Image):
            drawing = Image.fromarray(np.array(drawing))
        if drawing.mode != 'RGB':
            drawing = drawing.convert('RGB')

        # Ensure image is 512x512 as requested
        if drawing.size != (512, 512):
            print(f"Resizing input image from {drawing.size} to (512, 512)")
            processed_image = drawing.resize((512, 512), Image.Resampling.LANCZOS)
        else:
             processed_image = drawing

        # --- Prompt Preparation ---
        prompt = props.prompt
        negative_prompt = props.negative_prompt

        # Append defaults if needed (consider making these optional/configurable)
        # if not prompt.endswith("background."):
        #     prompt += " background."
        # if negative_prompt:
        #     negative_prompt += " NSFW, artifacts, blur, jpg, uncanny, deformed, glow, shadow."
        # else:
        #     negative_prompt = "NSFW, artifacts, blur, jpg, uncanny, deformed, glow, shadow."


        # --- LoRA Intensity (using cross_attention_kwargs as a potential channel) ---
        # Get current intensity if LoRA is loaded and scale was set
        cross_attention_kwargs = None # Default to None
        if self.lora_path and hasattr(self.pipeline, "cross_attention_kwargs") and self.pipeline.cross_attention_kwargs:
            # Use existing scale if already set by load_lora or set_lora_intensity
            cross_attention_kwargs = self.pipeline.cross_attention_kwargs
            current_lora_scale = cross_attention_kwargs.get("scale", 1.0) # For printing status
            print(f"Using LoRA with scale from cross_attention_kwargs: {current_lora_scale:.2f}")
        elif self.lora_path:
            # Fallback if somehow kwargs aren't set but lora_path exists
            print("Warning: LoRA path is set, but cross_attention_kwargs scale not found. Using default scale 1.0")
            cross_attention_kwargs = {"scale": 1.0}
            current_lora_scale = 1.0


        # --- Generation ---
        print(f"Generating image ({'SDXL' if self.is_sdxl else 'SD1.5'})...")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Negative Prompt: {negative_prompt[:100]}...")
        print(f"  Steps: {props.num_inference_steps}, CFG: {props.guidance_scale}, Img CFG: {props.image_guidance_scale}, Seed: {props.seed}")
        if self.lora_path and cross_attention_kwargs: # Update the print condition
            print(f"  LoRA: {os.path.basename(self.lora_path)}, Intensity: {current_lora_scale:.2f}")
            

        # Explicitly create generator on the correct device
        generator = None
        if props.seed is not None:
            generator = self._torch.Generator(device=self.device).manual_seed(props.seed)

        # Prepare kwargs for the pipeline call
        pipeline_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": processed_image,
            "num_inference_steps": props.num_inference_steps,
            "guidance_scale": props.guidance_scale,
            "image_guidance_scale": props.image_guidance_scale,
            "generator": generator, # Use the device-specific generator
            # Pass the determined kwargs, will be None if no LoRA active
            "cross_attention_kwargs": cross_attention_kwargs
        }


        # Add SDXL specific args if needed
        if self.is_sdxl:
            pipeline_kwargs["height"] = 512
            pipeline_kwargs["width"] = 512
            # SDXL InstructPix2Pix uses these micro-conditioning params
            pipeline_kwargs["original_size"] = (512, 512)
            pipeline_kwargs["target_size"] = (512, 512)
            pipeline_kwargs["crops_coords_top_left"] = (0, 0) # Default centering

        # --- Streaming Callback ---
        if callback:
            def callback_wrapper(step, timestep, latents):
                # Throttle callback frequency
                throttle_steps = max(1, props.num_inference_steps // 20) # Call ~20 times max
                if step % throttle_steps == 0 or step == props.num_inference_steps - 1:
                    try:
                        with self._torch.no_grad():
                             # Use the VAE's scaling factor - works for both SD1.5 & SDXL
                             latents_scaled = latents / self.pipeline.vae.config.scaling_factor
                             image = self.pipeline.vae.decode(latents_scaled).sample
                             image = (image / 2 + 0.5).clamp(0, 1)
                             image = image.cpu().permute(0, 2, 3, 1).numpy()[0] # Take first image in batch
                             image = (image * 255).astype(np.uint8)
                             pil_image = Image.fromarray(image)

                             # Convert to bytes more efficiently
                             import io
                             buffer = io.BytesIO()
                             pil_image.save(buffer, format="JPEG", quality=75) # Slightly lower quality for speed
                             img_bytes = buffer.getvalue()

                             # Call the provided callback
                             callback(step, props.num_inference_steps, img_bytes)
                    except Exception as cb_e:
                         print(f"Error in streaming callback at step {step}: {cb_e}")
                         # Optionally, stop streaming? For now, just print error.


            pipeline_kwargs["callback"] = callback_wrapper
            pipeline_kwargs["callback_steps"] = 1 # Call wrapper every step, throttling is internal

        if self.lora_path and cross_attention_kwargs: # Update the print condition
            print(f"  LoRA: {os.path.basename(self.lora_path)}, Intensity: {current_lora_scale:.2f}")


        with self._torch.no_grad():
             output_image = self.pipeline(**pipeline_kwargs).images[0]

        print("Image generation complete.")
        return output_image