from PIL import Image, ImageOps
import numpy as np
from types import SimpleNamespace
import os
from fidelity_mlp import FidelityMLP

class InferenceHandler:
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.schedulers = None
        self._setup_done = False
        
    def _lazy_setup(self):
        """Lazy load ML dependencies only when needed"""
        if self._setup_done:
            return
            
        import torch
        from diffusers import (
            StableDiffusionInstructPix2PixPipeline,
            DDIMScheduler,
            DDPMScheduler,
            PNDMScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.schedulers = {
            "DDIM": DDIMScheduler,
            "DDPM": DDPMScheduler,
            "PNDM": PNDMScheduler,
            "Euler Ancestral": EulerAncestralDiscreteScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler
        }
        
        self._torch = torch
        self._pipeline_class = StableDiffusionInstructPix2PixPipeline
        self._setup_done = True
        
    def get_scheduler_names(self):
        """Get list of available schedulers without loading ML stuff"""
        return ["DDIM", "DDPM", "PNDM", "Euler Ancestral", "DPM++ 2M"]
        
    def load_model(self, model_path, scheduler_name="DDIM"):
        """Load the model from path"""
        try:
            self._lazy_setup()
            
            scheduler_class = self.schedulers.get(scheduler_name)
            scheduler = scheduler_class.from_pretrained(model_path, subfolder="scheduler")
            
            self.pipeline = self._pipeline_class.from_pretrained(
                model_path,
                torch_dtype=self._torch.float16,
                safety_checker=None,
                scheduler=scheduler
            ).to(self.device)
            
            # Enable optimizations
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
    def load_model(self, model_path, scheduler_name="DDIM"):
        """Load the model from path"""
        try:
            self._lazy_setup()
            
            scheduler_class = self.schedulers.get(scheduler_name)
            scheduler = scheduler_class.from_pretrained(model_path, subfolder="scheduler")
            fidelity_mlp_path = model_path + "/fidelity_mlp"
            
            # Use from_pretrained_with_fidelity if fidelity_mlp_path is provided
            try:
                # First load the pipeline normally
                self.pipeline = self._pipeline_class.from_pretrained(
                    model_path,
                    torch_dtype=self._torch.float16,
                    safety_checker=None,
                    scheduler=scheduler
                ).to(self.device)
                
                # Then load and attach the fidelity MLP
                if os.path.exists(fidelity_mlp_path):
                    self.pipeline.fidelity_mlp = FidelityMLP.from_pretrained(fidelity_mlp_path)
                    self.pipeline.fidelity_mlp.to(device=self.device, dtype=self._torch.float16)
                    print(f"Loaded fidelity MLP from: {fidelity_mlp_path}")
                else:
                    print(f"Fidelity MLP path not found: {fidelity_mlp_path}")
                    return False
            except Exception as e:
                # Regular loading without fidelity MLP
                print(f"Error loading model: {e}")
                self.pipeline = self._pipeline_class.from_pretrained(
                    model_path,
                    torch_dtype=self._torch.float16,
                    safety_checker=None,
                    scheduler=scheduler
                ).to(self.device)
            
            # Enable optimizations
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def change_scheduler(self, scheduler_name):
        """Change scheduler if pipeline is loaded"""
        if self.pipeline and self._setup_done:
            scheduler_class = self.schedulers.get(scheduler_name)
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )
            
    def generate_image(self, drawing, props, callback=None):
        """Generate image from drawing"""
        if not self.pipeline:
            raise ValueError("Pipeline not loaded! Please load a model first.")
            
        # Convert drawing to PIL Image if needed
        if not isinstance(drawing, Image.Image):
            drawing = Image.fromarray(np.array(drawing))
            
        # Ensure image is in RGB mode
        if drawing.mode != 'RGB':
            drawing = drawing.convert('RGB')
            
        # Process drawing
        processed_image = drawing
        
        # Ensure image is the correct size
        if processed_image.size != (512, 512):
            processed_image = processed_image.resize((512, 512))
        print(props.prompt)
        
        if not props.prompt.endswith("background."):
            props.prompt += " background."
            
        props.negative_prompt=props.negative_prompt + "NSFW, artifacts, blur, jpg, uncanny, deformed, glow, shadow."
            
        # Generate image
        with self._torch.no_grad():
            if callback:  # If streaming is enabled
                # Create a custom callback that will be called at each step
                def callback_wrapper(step, timestep, latents):
                    # Only process every few steps to avoid overwhelming the UI
                    if step % max(1, props.num_inference_steps // 10) == 0 or step == props.num_inference_steps - 1:
                        # Decode the latents to get the intermediate image
                        with self._torch.no_grad():
                            latents_input = 1 / 0.18215 * latents
                            image = self.pipeline.vae.decode(latents_input).sample
                            image = (image / 2 + 0.5).clamp(0, 1)
                            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                            image = (image * 255).astype(np.uint8)
                            pil_image = Image.fromarray(image)
                            
                            # Convert to bytes for efficient transfer
                            import io
                            buffer = io.BytesIO()
                            pil_image.save(buffer, format="JPEG", quality=80)
                            img_bytes = buffer.getvalue()
                            
                            # Call the provided callback with step info and image bytes
                            callback(step, props.num_inference_steps, img_bytes)
                
                # Run the pipeline with the callback
                output = self.pipeline(
                    prompt=props.prompt,
                    negative_prompt=props.negative_prompt,
                    image=processed_image,
                    num_inference_steps=props.num_inference_steps,
                    guidance_scale=props.guidance_scale,
                    image_guidance_scale=props.image_guidance_scale,
                    generator=self._torch.manual_seed(props.seed) if props.seed else None,
                    callback=callback_wrapper,
                    callback_steps=1  # Process every step internally, but our wrapper will filter
                ).images[0]
            else:
                # Standard generation without streaming
                output = self.pipeline(
                    prompt=props.prompt,
                    negative_prompt=props.negative_prompt,
                    image=processed_image,
                    num_inference_steps=props.num_inference_steps,
                    guidance_scale=props.guidance_scale,
                    image_guidance_scale=props.image_guidance_scale,
                    generator=self._torch.manual_seed(props.seed) if props.seed else None
                ).images[0]
            
        return output