import torch
import torch.nn as nn
import os

class FidelityMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size  # Save for loading
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def save_pretrained(self, save_directory):
        """Save the model to a directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model config
        config = {
            "hidden_size": self.hidden_size
        }
        config_file = os.path.join(save_directory, "config.json")
        torch.save(config, config_file)
        
        # Save the model weights
        model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_file)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """Load the model from a directory"""
        config_file = os.path.join(pretrained_model_path, "config.json")
        model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
        
        # Load config
        config = torch.load(config_file)
        
        # Create model instance
        model = cls(hidden_size=config["hidden_size"])
        
        # Load weights
        model.load_state_dict(torch.load(model_file))
        return model


"""
Because the text encoder (CLIP) isn’t well adapted to learn a precise numeric token (like “f=5”) via tokenization, this module explicitly converts the extracted fidelity number into an embedding vector that can be directly processed by the UNet’s cross-attention layers.

2. Modify the Prompt Encoding in the Pipeline
In your pipeline code (in pipeline_stable_diffusion_instruct_pix2pix.py), locate the _encode_prompt method. You’ll add code to extract the fidelity value from the prompt and then prepend the corresponding fidelity embedding to the text embeddings.

For example, modify _encode_prompt as follows (insert the new code block after obtaining prompt_embeds):

python
Copy
def _encode_prompt(
    self,
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    return_teacher_loss: bool = False,
):
    # (existing code to tokenize and get prompt_embeds)
    if prompt_embeds is None:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device) if hasattr(self.text_encoder.config, "use_attention_mask") else None
        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=attention_mask)[0]

    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
    
    # --- New Fidelity Injection ---
    # Extract fidelity value from the prompt (assumes prompt starts with something like "f=5")
    import re
    fidelity_val = 0.5  # default fidelity
    if isinstance(prompt, str):
        match = re.search(r"f\s*=?\s*(\d+)", prompt, re.IGNORECASE)
        if match:
            f_int = int(match.group(1))
            f_int = max(1, min(f_int, 9))
            fidelity_val = 0.1 + (f_int - 1) * (0.8 / 8)
    elif isinstance(prompt, list):
        # For a list, average the fidelity values extracted from each prompt.
        f_vals = []
        for p in prompt:
            match = re.search(r"f\s*=?\s*(\d+)", p, re.IGNORECASE)
            if match:
                f_int = int(match.group(1))
                f_int = max(1, min(f_int, 9))
                f_vals.append(0.1 + (f_int - 1) * (0.8 / 8))
        if f_vals:
            fidelity_val = sum(f_vals) / len(f_vals)
    batch_size = prompt_embeds.shape[0]
    fidelity_tensor = torch.full((batch_size, 1), fidelity_val, device=device, dtype=prompt_embeds.dtype)
    
    # Here, assume that your pipeline instance has an attribute fidelity_mlp, set at initialization.
    fidelity_embedding = self.fidelity_mlp(fidelity_tensor)  # (batch, hidden_size)
    fidelity_embedding = fidelity_embedding.unsqueeze(1)  # (batch, 1, hidden_size)
    
    # Prepend the fidelity embedding to the text embeddings
    prompt_embeds = torch.cat([fidelity_embedding, prompt_embeds], dim=1)
    # --- End Fidelity Injection ---
    
    # Duplicate embeddings for num_images_per_prompt, etc.
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs_embed * num_images_per_prompt, seq_len, -1)
    return prompt_embeds
Why?
By prepending a fidelity token (learned via fidelity_mlp) to the sequence, you explicitly inject the fidelity condition into the conditioning that is used by the UNet’s cross-attention. The network can then learn to modulate its behavior based on this signal.

3. Integrate FidelityMLP into Your UNet Training Script
Since you train your UNet with the text encoder frozen (text encoder is loaded separately and not updated), you want to update only the new fidelity module and the UNet parameters. In your train_instruct_pix2pix.py script, do the following modifications:

Instantiate and attach the FidelityMLP:
In the section where you load the text encoder and before you prepare your pipeline, add:

python
Copy
from fidelity_mlp import FidelityMLP

# Load text encoder (already done)
text_encoder = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="text_encoder",
    revision=args.revision,
    variant=args.variant,
)
# Freeze the text encoder
text_encoder.requires_grad_(False)

# Instantiate FidelityMLP with the same hidden size as the text encoder.
hidden_size = text_encoder.config.hidden_size
fidelity_mlp = FidelityMLP(hidden_size)
# You may want to move fidelity_mlp to device:
fidelity_mlp.to(accelerator.device)
Attach the fidelity_mlp to the pipeline’s text encoder object:
In the pipeline code, you can set an attribute so that the _encode_prompt function can access it:

python
Copy
# After preparing the pipeline, do:
pipeline.fidelity_mlp = fidelity_mlp
Or, if you are directly calling the _encode_prompt method from your training loop, ensure that the fidelity_mlp is accessible.

Optimizer Setup:
Since the text encoder is frozen, you do not pass its parameters to the optimizer. Instead, add only the fidelity MLP’s parameters along with the UNet (and any other modules you want to update). For example:

python
Copy
params_to_optimize = [
    {"params": unet.parameters(), "lr": args.learning_rate},
    {"params": fidelity_mlp.parameters(), "lr": args.fidelity_learning_rate if hasattr(args, "fidelity_learning_rate") else args.learning_rate}
]
optimizer = torch.optim.AdamW(params_to_optimize, betas=(args.adam_beta1, args.adam_beta2),
                              weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
Note: Adjust your command-line arguments or hardcode a learning rate for the fidelity MLP if needed.

Why?
You want the text encoder to remain fixed (since it’s been trained separately), and only the new fidelity signal should be learned jointly with the UNet. This decouples the numeric fidelity conditioning from the rest of the text conditioning and gives a stronger, dedicated signal to guide image editing.

4. Training and Inference Impact
Expected Improvements:

Stronger Fidelity Signal:
By explicitly providing a learned embedding for fidelity, the UNet’s cross-attention layers receive a clear numeric cue. This should help the model generate images that either adhere very strictly to the input (high fidelity) or allow more creative deviation (low fidelity), as per your dataset examples.
Decoupled Conditioning:
The text encoder remains frozen, and the fidelity MLP is lightweight and dedicated solely to this task. This separation should make it easier for the network to learn how to interpret the fidelity value.
Flexibility at Inference:
Once trained, you can control fidelity simply by modifying the extracted scalar or even overriding it at inference time, and the UNet will have learned to respond accordingly.
Summary
Create a FidelityMLP module (see fidelity_mlp.py).
Modify _encode_prompt in your pipeline to:
Extract a fidelity value from the prompt.
Pass it through fidelity_mlp to create an embedding.
Prepend this embedding to the text encoder’s output.
In your UNet training script, freeze the text encoder and update the optimizer to include only the UNet and the new fidelity module.
Train the model. The UNet will now condition on an explicit fidelity signal, which should help it learn to modulate output creativity and adherence.
"""