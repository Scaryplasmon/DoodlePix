import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from peft import PeftModel
import re
import logging
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

class TextEncoderValidator:
    def __init__(self, base_model_path: str, finetuned_model_path: str = None):
        # Load base model components from the pipeline
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16
        )
        
        self.tokenizer = pipe.tokenizer
        self.base_encoder = pipe.text_encoder.to(self.device)
        
        # Load fine-tuned LoRA model if provided
        if finetuned_model_path:
            self.finetuned_encoder = PeftModel.from_pretrained(
                self.base_encoder,
                finetuned_model_path
            ).to(self.device)
        else:
            self.finetuned_encoder = None
        
        # Test cases matching our prompt structure
        self.test_cases = {
            'subject_color_pairs': [
                ('egg', 'rock (gray), eggshell (white)'),
                ('orb', 'crystal (blue), glass (clear)'),
                ('shield', 'metal (silver), wood (brown)'),
                ('sword', 'steel (gray), leather (brown)'),
            ],
            'worlds': ['earth', 'fantasy', 'sci-fi', 'medieval', 'steampunk'],
            'styles': ['3d', 'flat', 'pixel', 'realistic', 'hand drawn'],
            'detail_levels': [1, 2, 3, 4, 5]
        }

    def get_embedding(self, text: str, model: CLIPTextModel) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**tokens)
                return output.pooler_output

    def validate_subject_color_consistency(self, model: CLIPTextModel):
        """Validate subject-color relationship understanding"""
        results = {}
        
        for subject, colors in self.test_cases['subject_color_pairs']:
            # Create full prompt
            prompt = f"<s:{subject}>, <c:{colors}>, <w:earth>, <k:3>, <p:3d>"
            base_emb = self.get_embedding(prompt, model)
            
            # Test with different styles
            style_embeddings = []
            for style in self.test_cases['styles']:
                style_prompt = f"<s:{subject}>, <c:{colors}>, <w:earth>, <k:3>, <p:{style}>"
                style_emb = self.get_embedding(style_prompt, model)
                style_embeddings.append(style_emb)
            
            # Calculate consistency scores
            style_embeddings = torch.cat(style_embeddings, dim=0)
            sim_matrix = cosine_similarity(style_embeddings.cpu().numpy())
            
            results[subject] = {
                'mean_consistency': float(np.mean(sim_matrix)),
                'std_consistency': float(np.std(sim_matrix))
            }
            
        return results

    def validate_complexity_progression(self, model: CLIPTextModel):
        """Validate detail level understanding"""
        results = {}
        base_prompt = "<s:orb>, <c:crystal (blue)>, <w:fantasy>, <p:3d>"
        
        # Get embeddings for each detail level
        detail_embeddings = []
        for level in self.test_cases['detail_levels']:
            prompt = f"{base_prompt}, <k:{level}>"
            emb = self.get_embedding(prompt, model)
            detail_embeddings.append(emb)
        
        # Calculate progression
        detail_embeddings = torch.cat(detail_embeddings, dim=0)
        sim_matrix = cosine_similarity(detail_embeddings.cpu().numpy())
        
        # Check if higher detail levels are more different
        progression_scores = []
        for i in range(len(sim_matrix)-1):
            progression_scores.append(1 - sim_matrix[i, i+1])
        
        results['progression_scores'] = progression_scores
        results['monotonic_increase'] = all(x < y for x, y in zip(progression_scores, progression_scores[1:]))
        
        return results

    def validate_style_world_compatibility(self, model: CLIPTextModel):
        """Validate style-world relationship understanding"""
        results = {}
        
        for world in self.test_cases['worlds']:
            style_scores = {}
            for style in self.test_cases['styles']:
                prompt = f"<s:orb>, <c:crystal (blue)>, <w:{world}>, <k:3>, <p:{style}>"
                emb = self.get_embedding(prompt, model)
                
                # Test compatibility with world context
                world_context = f"a {world} themed UI element"
                context_emb = self.get_embedding(world_context, model)
                
                compatibility = float(F.cosine_similarity(emb, context_emb, dim=1))
                style_scores[style] = compatibility
                
            results[world] = style_scores
        
        return results

    def validate_with_dataset_samples(self, model: CLIPTextModel, data_path: str, num_samples: int = 10):
        """Compare embeddings of actual dataset prompts"""
        results = {}
        
        # Load random prompts from dataset
        txt_files = list(Path(data_path).glob('*.txt'))
        selected_files = np.random.choice(txt_files, min(num_samples, len(txt_files)), replace=False)
        
        for txt_file in selected_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                
            # Get embedding
            emb = self.get_embedding(prompt, model)
            
            # Store results
            results[txt_file.name] = {
                'prompt': prompt,
                'embedding_norm': float(torch.norm(emb).cpu()),
                'embedding_mean': float(torch.mean(emb).cpu()),
                'embedding_std': float(torch.std(emb).cpu())
            }
        
        return results

    def compare_embeddings(self, prompt: str):
        """Directly compare base and finetuned embeddings for a single prompt"""
        base_emb = self.get_embedding(prompt, self.base_encoder)
        finetuned_emb = self.get_embedding(prompt, self.finetuned_encoder)
        
        # Calculate differences
        cosine_sim = F.cosine_similarity(base_emb, finetuned_emb).cpu().item()
        l2_dist = torch.norm(base_emb - finetuned_emb).cpu().item()
        
        return {
            'cosine_similarity': cosine_sim,
            'l2_distance': l2_dist
        }

    def plot_results(self, results: dict, output_dir: str):
        """Plot validation results"""
        # Plot subject-color consistency
        plt.figure(figsize=(10, 6))
        subjects = list(results['subject_color_consistency'].keys())
        consistency_scores = [results['subject_color_consistency'][s]['mean_consistency'] 
                            for s in subjects]
        
        plt.bar(subjects, consistency_scores)
        plt.title('Subject-Color Consistency Across Styles')
        plt.ylabel('Consistency Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/subject_color_consistency.png')
        plt.close()
        
        # Plot complexity progression
        plt.figure(figsize=(10, 6))
        plt.plot(results['complexity_progression']['progression_scores'], marker='o')
        plt.title('Detail Level Progression')
        plt.xlabel('Detail Level Transition')
        plt.ylabel('Difference Score')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/complexity_progression.png')
        plt.close()
        
        # Plot style-world compatibility
        plt.figure(figsize=(12, 8))
        data = []
        for world, style_scores in results['style_world_compatibility'].items():
            for style, score in style_scores.items():
                data.append({'World': world, 'Style': style, 'Compatibility': score})
        
        compatibility_matrix = np.zeros((len(self.test_cases['worlds']), 
                                      len(self.test_cases['styles'])))
        for i, world in enumerate(self.test_cases['worlds']):
            for j, style in enumerate(self.test_cases['styles']):
                compatibility_matrix[i, j] = results['style_world_compatibility'][world][style]
        
        sns.heatmap(compatibility_matrix, 
                   xticklabels=self.test_cases['styles'],
                   yticklabels=self.test_cases['worlds'],
                   cmap='YlOrRd')
        plt.title('Style-World Compatibility')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/style_world_compatibility.png')
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Lykon/dreamshaper-8")
    parser.add_argument("--finetuned_model", type=str, default="./DoodlePixTxtEncoderV3")
    parser.add_argument("--data_path", type=str, default="DoodlePixV4/edit_prompt/")
    parser.add_argument("--output_dir", type=str, default="validation_results")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    validator = TextEncoderValidator(args.base_model, args.finetuned_model)
    
    # Test with dataset samples
    dataset_results = validator.validate_with_dataset_samples(
        validator.finetuned_encoder, 
        args.data_path
    )
    
    # Compare embeddings for each sample
    comparison_results = {}
    for filename, data in dataset_results.items():
        comparison_results[filename] = validator.compare_embeddings(data['prompt'])
    
    # Save detailed results
    with open(f'{args.output_dir}/dataset_samples.json', 'w') as f:
        json.dump(dataset_results, f, indent=2)
    
    with open(f'{args.output_dir}/embedding_comparisons.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Print summary
    print("\nEmbedding Comparison Summary:")
    cosine_sims = [r['cosine_similarity'] for r in comparison_results.values()]
    l2_dists = [r['l2_distance'] for r in comparison_results.values()]
    
    print(f"Average Cosine Similarity: {np.mean(cosine_sims):.4f}")
    print(f"Average L2 Distance: {np.mean(l2_dists):.4f}")
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
