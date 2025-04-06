# fidelity_mlp.py
import torch
import torch.nn as nn
import os

class FidelityMLP(nn.Module):
    def __init__(self, hidden_size, output_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),  # Bound outputs between -1 and 1
        )
        
        self.output_proj = nn.Linear(hidden_size, self.output_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use small initial weights
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x, target_dim=None):
        """
        Forward pass with optional target dimension adjustment
        
        Args:
            x: Input tensor with shape [batch_size, 1]
            target_dim: If provided, will project to this dimension instead
        """
        # Process through main network
        features = self.net(x)
        
        # Project to embedding space
        outputs = self.output_proj(features)
        
        # If target dimension is provided and different than output_size,
        # we need to adjust the dimensionality on the fly
        if target_dim is not None and target_dim != self.output_size:
            return self._adjust_dimension(outputs, target_dim)
            
        return outputs
    
    def _adjust_dimension(self, embeddings, target_dim):
        """
        Adjusts embedding dimension to match target_dim
        
        This allows the model to handle any embedding size on the fly
        """
        current_dim = embeddings.shape[-1]
        
        # If target is larger, pad with zeros
        if target_dim > current_dim:
            pad_size = target_dim - current_dim
            # Create padding for last dimension only
            padding = torch.zeros(
                (*embeddings.shape[:-1], pad_size),
                device=embeddings.device, 
                dtype=embeddings.dtype
            )
            return torch.cat([embeddings, padding], dim=-1)
            
        # If target is smaller, truncate
        elif target_dim < current_dim:
            return embeddings[..., :target_dim]
            
        # If same size, return as is
        return embeddings
    
    def save_pretrained(self, save_directory):
        """Save the model to a directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "hidden_size": self.hidden_size,
            "output_size": self.output_size
        }
        config_file = os.path.join(save_directory, "config.json")
        torch.save(config, config_file)
        
        model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_file)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """Load the model from a directory"""
        config_file = os.path.join(pretrained_model_path, "config.json")
        model_file = os.path.join(pretrained_model_path, "pytorch_model.bin")
        
        config = torch.load(config_file)
        
        model = cls(
            hidden_size=config["hidden_size"],
            output_size=config.get("output_size", config["hidden_size"])
        )
        
        model.load_state_dict(torch.load(model_file))
        return model
