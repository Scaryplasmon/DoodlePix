import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

class StyleMaterialInteractionLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, style_emb, material_emb, material_type, style_type):
        # Compute similarity matrix for the batch
        sim_matrix = torch.matmul(style_emb, material_emb.t()) / self.temperature
        
        # Create labels based on compatible style-material pairs
        labels = self._get_compatibility_matrix(material_type, style_type)
        
        # Ensure labels match batch size
        if labels.size(0) != style_emb.size(0):
            labels = labels[:style_emb.size(0), :style_emb.size(0)]
        
        # Reshape if needed
        if sim_matrix.size() != labels.size():
            sim_matrix = sim_matrix[:style_emb.size(0), :style_emb.size(0)]
        
        return F.cross_entropy(sim_matrix, labels)
    
    def _get_compatibility_matrix(self, material_types, style_types):
        compatibility = {
            'metal': ['3d', 'realistic'],
            'glass': ['3d', 'realistic'],
            'wood': ['flat', 'realistic', 'hand drawn'],
            'crystal': ['3d', 'realistic'],
            'stone': ['flat', 'realistic'],
            'fabric': ['flat', 'hand drawn', 'realistic']
        }
        return self._create_compatibility_labels(material_types, style_types, compatibility)
    
    def _create_compatibility_labels(self, material_types, style_types, compatibility_dict):
        batch_size = len(material_types)
        device = material_types[0].device if hasattr(material_types[0], 'device') else 'cuda'
        labels = torch.zeros((batch_size, batch_size), device=device)
        
        for i, mat in enumerate(material_types):
            for j, style in enumerate(style_types):
                try:
                    mat_name = re.match(r'([^(]+)', mat).group(1).strip()
                    if mat_name in compatibility_dict and style in compatibility_dict[mat_name]:
                        labels[i, j] = 1.0
                except (AttributeError, TypeError):
                    continue
        
        # Normalize rows
        row_sums = labels.sum(dim=1, keepdim=True)
        labels = labels / (row_sums + 1e-6)
        
        return labels

class ComplexityGuidanceLoss(nn.Module):
    def __init__(self, num_levels=5):
        super().__init__()
        self.num_levels = num_levels
        
    def forward(self, embeddings, levels):
        level_centroids = []
        for i in range(1, self.num_levels + 1):
            mask = (levels == i)
            if mask.any():
                level_centroids.append(embeddings[mask].mean(0))
        
        loss = 0
        for i in range(len(level_centroids) - 1):
            complexity_diff = F.cosine_similarity(level_centroids[i], level_centroids[i+1], dim=0)
            progression_loss = torch.relu(complexity_diff - 0.8)  # Expect at least 0.2 difference
            loss += progression_loss
            
        return loss

class MaterialColorConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, material_emb, color_emb, material_type):
        material_centroid = material_emb.mean(0, keepdim=True)
        material_variance = F.mse_loss(material_emb, material_centroid.expand_as(material_emb))
        color_influence = F.cosine_similarity(material_emb, color_emb).mean()
        return material_variance + (1.0 - color_influence) * 0.5

class CrossComponentAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, 8)
        
    def forward(self, components_dict):
        keys = ['style', 'colors', 'world', 'description']
        try:
            device = next(self.attention.parameters()).device
            embeddings = torch.stack([components_dict[k].to(device) for k in keys])
            attn_output, _ = self.attention(embeddings, embeddings, embeddings)
            return attn_output
        except KeyError as e:
            logger.error(f"Missing component key: {e}")
            raise

class CustomTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length
        
        # Load all .txt files
        txt_files = list(Path(data_path).glob('*.txt'))
        for txt_path in txt_files:
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                if prompt:
                    self.data.append(prompt)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt = self.data[idx]
        
        # Tokenize full prompt
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Parse components
        components = {}
        for tag in ['s', 'c', 'w', 'k', 'p', 'd']:
            pattern = f'<{tag}:([^>]+)>'
            match = re.search(pattern, prompt)
            if match:
                content = match.group(1).strip()
                comp_tokens = self.tokenizer(
                    content,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                components[tag] = {
                    'content': content,
                    'input_ids': comp_tokens['input_ids'].squeeze(0),
                    'attention_mask': comp_tokens['attention_mask'].squeeze(0)
                }
        
        # Extract detail level
        detail_level = 3  # default
        if 'k' in components:
            try:
                detail_level = int(components['k']['content'])
            except ValueError:
                pass
        
        return {
            'prompt': prompt,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'components': components,
            'detail_level': torch.tensor(detail_level, dtype=torch.long)
        }

def train_text_encoder(
    pretrained_model_name: str,
    train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    max_train_steps: int = 1000,
    output_dir: str = "text_encoder_finetuned",
    data_path: str = "./prompts",
):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    
    # Load models
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float16
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    
    # Initialize loss modules
    style_material_loss = StyleMaterialInteractionLoss().to(accelerator.device)
    complexity_loss = ComplexityGuidanceLoss().to(accelerator.device)
    material_color_loss = MaterialColorConsistencyLoss().to(accelerator.device)
    cross_attention = CrossComponentAttention(text_encoder.config.hidden_size).to(accelerator.device)
    
    # Setup dataset and dataloader
    dataset = CustomTextDataset(data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True
    )
    
    optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=learning_rate)
    
    text_encoder, optimizer, dataloader = accelerator.prepare(
        text_encoder, optimizer, dataloader
    )
    
    global_step = 0
    losses = []
    progress_bar = tqdm(total=max_train_steps)
    
    text_encoder.train()
    while global_step < max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(text_encoder):
                # Process full prompt
                outputs = text_encoder(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=True
                )
                
                # Process components
                component_embeddings = {}
                for tag, comp_data in batch['components'].items():
                    comp_output = text_encoder(
                        input_ids=comp_data['input_ids'],
                        attention_mask=comp_data['attention_mask'],
                        return_dict=True
                    )
                    component_embeddings[tag] = comp_output.pooler_output
                
                # Calculate losses
                style_mat_loss = style_material_loss(
                    component_embeddings['p'],
                    component_embeddings['c'],
                    [batch['components']['c']['content']],
                    [batch['components']['p']['content']]
                )
                
                comp_loss = complexity_loss(
                    outputs.pooler_output,
                    batch['detail_level']
                )
                
                mat_color_loss = material_color_loss(
                    component_embeddings['c'],
                    component_embeddings['p'],
                    [batch['components']['c']['content']]
                )
                
                # Cross-component attention
                cross_attn = cross_attention(component_embeddings)
                comp_stack = torch.stack([
                    component_embeddings[k] for k in ['p', 'c', 'w', 'd']
                ])
                cross_attn_loss = F.mse_loss(cross_attn, comp_stack)
                
                # Combined loss
                loss = (
                    2.0 * style_mat_loss +
                    1.5 * comp_loss +
                    1.5 * mat_color_loss +
                    1.0 * cross_attn_loss
                )
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(text_encoder.parameters(), 1.0)
                    
                optimizer.step()
                optimizer.zero_grad()
                
                losses.append(loss.item())
                
                if global_step % 100 == 0:
                    avg_loss = sum(losses[-100:]) / len(losses[-100:])
                    logger.info(f"Step {global_step}: Average loss = {avg_loss:.4f}")
                
                global_step += 1
                progress_bar.update(1)
                
                if global_step >= max_train_steps:
                    break
    
    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(text_encoder)
    unwrapped_model.save_pretrained(output_dir)
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()
    
    # Save loss values
    with open(os.path.join(output_dir, 'training_loss.json'), 'w') as f:
        json.dump({'losses': losses}, f)
    
    return text_encoder

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type=str, default="Lykon/dreamshaper-8")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    train_text_encoder(
        pretrained_model_name=args.pretrained_model_name,
        data_path=args.data_path,
        output_dir=args.output_dir
    )