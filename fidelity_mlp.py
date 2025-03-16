import torch
import torch.nn as nn
import os

class FidelityMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
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
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use small initial weights
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        # Process through main network
        features = self.net(x)
        
        # Project to embedding space
        # No scaling here - we'll let the model learn the appropriate scale
        return self.output_proj(features)
    
    def save_pretrained(self, save_directory):
        """Save the model to a directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "hidden_size": self.hidden_size
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
        
        model = cls(hidden_size=config["hidden_size"])
        
        model.load_state_dict(torch.load(model_file))
        return model
