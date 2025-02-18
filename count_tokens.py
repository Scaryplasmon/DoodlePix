from transformers import CLIPTokenizer
import json
from pathlib import Path

def count_tokens(text, tokenizer):
    """Count tokens in a text string"""
    tokens = tokenizer(text)['input_ids']
    return len(tokens)

def get_truncated_text(text, tokenizer):
    """Get the text that will be truncated"""
    tokens = tokenizer(text)['input_ids']
    if len(tokens) <= 77:
        return None
        
    # Decode the tokens that will be truncated
    truncated_tokens = tokens[77:]
    truncated_text = tokenizer.decode(truncated_tokens)
    return truncated_text.strip()

def analyze_prompt_tokens(prompt_file, tokenizer):
    """Analyze token count in a prompt file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    token_count = count_tokens(text, tokenizer)
    is_truncated = token_count > 77
    
    result = {
        'total_tokens': token_count,
        'will_be_truncated': is_truncated,
        'tokens_over_limit': token_count - 77 if is_truncated else 0
    }
    
    if is_truncated:
        truncated_text = get_truncated_text(text, tokenizer)
        result['truncated_text'] = truncated_text
        
    return result

def batch_analyze_prompts(directory):
    """Analyze all .txt files in a directory"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    results = {}
    txt_files = Path(directory).glob('*.txt')
    
    for txt_file in txt_files:
        try:
            analysis = analyze_prompt_tokens(txt_file, tokenizer)
            results[txt_file.name] = analysis
            
            print(f"\nFile: {txt_file.name}")
            print(f"Total tokens: {analysis['total_tokens']}")
            if analysis['will_be_truncated']:
                print(f"WARNING: Will be truncated! Over by {analysis['tokens_over_limit']} tokens")
                print(f"Truncated text: {analysis['truncated_text']}")
            else:
                print("OK: Within token limit")
                
        except Exception as e:
            print(f"Error processing {txt_file.name}: {str(e)}")
    
    # Save results
    with open('token_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze token counts in prompt files')
    # parser.add_argument('directory', help='Directory containing .txt prompt files', default="countTokens")
    args = parser.parse_args()
    directory = "DoodlePixV4/DoodlePixV5/A_3D/edit_prompt/"
    
    batch_analyze_prompts(directory)