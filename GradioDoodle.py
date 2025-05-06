# --- Fix 1: Set Matplotlib backend ---
import matplotlib
matplotlib.use('Agg') # Set backend BEFORE importing pyplot or other conflicting libs
# --- End Fix 1 ---

import gradio as gr
import torch
from diffusers import  EulerAncestralDiscreteScheduler
from DoodlePix_pipeline import StableDiffusionInstructPix2PixPipeline
from PIL import Image, ImageOps # Added ImageOps for inversion
import numpy as np
import os
import importlib
import traceback # For detailed error printing

# --- FidelityMLP Class (Ensure this is correct as provided by user) ---
class FidelityMLP(torch.nn.Module):
    def __init__(self, hidden_size, output_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 128), torch.nn.LayerNorm(128), torch.nn.SiLU(),
            torch.nn.Linear(128, 256), torch.nn.LayerNorm(256), torch.nn.SiLU(),
            torch.nn.Linear(256, hidden_size), torch.nn.LayerNorm(hidden_size), torch.nn.Tanh()
        )
        self.output_proj = torch.nn.Linear(hidden_size, self.output_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None: module.bias.data.zero_()

    def forward(self, x, target_dim=None):
        features = self.net(x)
        outputs = self.output_proj(features)
        if target_dim is not None and target_dim != self.output_size:
            return self._adjust_dimension(outputs, target_dim)
        return outputs

    def _adjust_dimension(self, embeddings, target_dim):
        current_dim = embeddings.shape[-1]
        if target_dim > current_dim:
            pad_size = target_dim - current_dim
            padding = torch.zeros((*embeddings.shape[:-1], pad_size), device=embeddings.device, dtype=embeddings.dtype)
            return torch.cat([embeddings, padding], dim=-1)
        elif target_dim < current_dim:
            return embeddings[..., :target_dim]
        return embeddings

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config = {"hidden_size": self.hidden_size, "output_size": self.output_size}
        torch.save(config, os.path.join(save_directory, "config.json"))
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        config_file = os.path.join(pretrained_model_path, "config.json")
        model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
        if not os.path.exists(config_file): raise FileNotFoundError(f"Config file not found at {config_file}")
        if not os.path.exists(model_file): raise FileNotFoundError(f"Model file not found at {model_file}")
        try:
            config = torch.load(config_file, map_location=torch.device('cpu'))
            if not isinstance(config, dict): raise TypeError(f"Expected config dict, got {type(config)}")
        except Exception as e: print(f"Error loading config {config_file}: {e}"); raise
        model = cls(hidden_size=config["hidden_size"], output_size=config.get("output_size", config["hidden_size"]))
        try:
            state_dict = torch.load(model_file, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"Successfully loaded FidelityMLP state dict from {model_file}")
        except Exception as e: print(f"Error loading state dict {model_file}: {e}"); raise
        return model

# --- Global Variables ---
pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Scaryplasmon96/DoodlePixV1"

# --- Model Loading Function ---
def load_pipeline():
    global pipeline
    if pipeline is not None: return True
    print(f"Loading model {model_id} onto {device}...")
    try:
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        local_model_path = model_id # Let diffusers find/download

        # Load Fidelity MLP if possible
        fidelity_mlp_instance = None
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            # Attempt to download config first to check existence
            hf_hub_download(repo_id=model_id, filename="fidelity_mlp/config.json", cache_dir=hf_cache_dir)
            # If config exists, download the whole subfolder
            fidelity_mlp_path = snapshot_download(repo_id=model_id, allow_patterns="fidelity_mlp/*", local_dir_use_symlinks=False, cache_dir=hf_cache_dir)
            fidelity_mlp_instance = FidelityMLP.from_pretrained(os.path.join(fidelity_mlp_path, "fidelity_mlp"))
            fidelity_mlp_instance = fidelity_mlp_instance.to(device=device, dtype=torch.float16)
            print("Fidelity MLP loaded successfully.")
        except Exception as e:
            print(f"Fidelity MLP not found or failed to load for {model_id}: {e}. Proceeding without MLP.")
            fidelity_mlp_instance = None

        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(local_model_path, subfolder="scheduler")
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            local_model_path, torch_dtype=torch.float16, scheduler=scheduler, safety_checker=None
        ).to(device)

        if fidelity_mlp_instance:
            pipeline.fidelity_mlp = fidelity_mlp_instance
            print("Attached Fidelity MLP to pipeline.")

        # Optimizations
        if device == "cuda" and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try: pipeline.enable_xformers_memory_efficient_attention(); print("Enabled xformers.")
            except: print("Could not enable xformers. Using attention slicing."); pipeline.enable_attention_slicing()
        else: pipeline.enable_attention_slicing(); print("Enabled attention slicing.")

        print("Pipeline loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading pipeline: {e}"); traceback.print_exc()
        pipeline = None; raise gr.Error(f"Failed to load model: {e}")

# --- Image Generation Function (Corrected Input Handling) ---
def generate_image(drawing_input, prompt, fidelity_slider, steps, guidance, image_guidance, seed_val):
    global pipeline
    if pipeline is None:
        if not load_pipeline(): return None, "Model not loaded. Check logs."

    # --- Corrected Input Processing ---
    print(f"DEBUG: Received drawing_input type: {type(drawing_input)}")
    if isinstance(drawing_input, dict): print(f"DEBUG: Received drawing_input keys: {drawing_input.keys()}")

    # Check if input is dict and get PIL image from 'composite' key
    if isinstance(drawing_input, dict) and "composite" in drawing_input and isinstance(drawing_input["composite"], Image.Image):
        input_image_pil = drawing_input["composite"].convert("RGB") # Get composite image
        print("DEBUG: Using PIL Image from 'composite' key.")
    else:
        err_msg = "Drawing input format unexpected. Expected dict with PIL Image under 'composite' key."
        print(f"ERROR: {err_msg} Input: {drawing_input}")
        return None, err_msg
    # --- End Corrected Input Processing ---

    try:
        # Invert the image: White bg -> Black bg, Black lines -> White lines
        input_image_inverted = ImageOps.invert(input_image_pil)
        #save the inverted image
        # input_image_inverted.save("input_image_inverted.png")

        # Ensure image is 512x512
        if input_image_inverted.size != (512, 512):
            print(f"Resizing input image from {input_image_inverted.size} to (512, 512)")
            input_image_inverted = input_image_inverted.resize((512, 512), Image.Resampling.LANCZOS)

        # Prompt Construction
        final_prompt = f"f{int(fidelity_slider)}, {prompt}"
        if not final_prompt.endswith("background."): final_prompt += " background."

        negative_prompt = "artifacts, blur, jpg, uncanny, deformed, glow, shadow, text, words, letters, signature, watermark"

        # Generation
        print(f"Generating with: Prompt='{final_prompt[:100]}...', Fidelity={int(fidelity_slider)}, Steps={steps}, Guidance={guidance}, ImageGuidance={image_guidance}, Seed={seed_val}")
        seed_val = int(seed_val)
        generator = torch.Generator(device=device).manual_seed(seed_val)

        with torch.no_grad():
             output = pipeline(
                 prompt=final_prompt, negative_prompt=negative_prompt, image=input_image_inverted,
                 num_inference_steps=int(steps), guidance_scale=float(guidance),
                 image_guidance_scale=float(image_guidance), generator=generator,
             ).images[0]

        print("Generation complete.")
        return output, "Generation Complete"

    except Exception as e:
        print(f"Error during generation: {e}"); traceback.print_exc()
        return None, f"Error during generation: {str(e)}"

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue")) as demo:
    gr.Markdown("# DoodlePix Gradio App")
    gr.Markdown(f"Using model: `{model_id}`.")
    status_output = gr.Textbox(label="Status", interactive=False, value="App loading...")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Draw Something (Black on White)")
            # Keep type="pil" as it provides the composite key
            drawing = gr.Sketchpad(
                label="Drawing Canvas",
                type="pil", # type="pil" gives dict output with 'composite' key
                height=512, width=512,
                brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=5),
                show_label=True
            )
            prompt_input = gr.Textbox(label="2. Enter Prompt", placeholder="Describe the image you want...")
            fidelity = gr.Slider(0, 9, step=1, value=4, label="Fidelity (0=Creative, 9=Faithful)")
            num_steps = gr.Slider(10, 50, step=1, value=25, label="Inference Steps")
            guidance_scale = gr.Slider(1.0, 15.0, step=0.5, value=7.5, label="Guidance Scale (CFG)")
            image_guidance_scale = gr.Slider(0.5, 5.0, step=0.1, value=1.5, label="Image Guidance Scale")
            seed = gr.Number(label="Seed", value=42, precision=0)
            generate_button = gr.Button("🚀 Generate Image!", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## 3. Generated Image")
            output_image = gr.Image(label="Result", type="pil", height=512, width=512, show_label=True)

    generate_button.click(
        fn=generate_image,
        inputs=[drawing, prompt_input, fidelity, num_steps, guidance_scale, image_guidance_scale, seed],
        outputs=[output_image, status_output]
    )

# --- Launch App ---
if __name__ == "__main__":
    initial_status = "App loading..."
    print("Attempting to pre-load pipeline...")
    try:
        if load_pipeline(): initial_status = "Model pre-loaded successfully."
        else: initial_status = "Model pre-loading failed. Will retry on first generation."
    except Exception as e:
        print(f"Pre-loading failed: {e}")
        initial_status = f"Model pre-loading failed: {e}. Will retry on first generation."
    print(f"Pre-loading status: {initial_status}")

    demo.launch()