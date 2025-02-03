from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
import json
from pathlib import Path
#4.41.2
# revision = "2024-08-26"  # Pin to specific version
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, trust_remote_code=True, revision=revision,
#     torch_dtype=torch.float16, attn_implementation="eager"
# ).to("cuda")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)


def get_structured_description(image):
    # Get main subject (short and concise)
        
    main_subject = model.query(image, "What is the main subject of this image? max 3 words")["answer"]
    
    theme = model.query(image, "Describe the theme of this icon? max 3 words")["answer"]
    
    colors = model.query(image, "Which 3 colors are the most used for the icon? give back the answer in this format, add the part relative to that color within the answer: <colorfg1>, <colorfg2>, <colorfg3>.")["answer"]
    
    materials = model.query(image, "Which 3 materials and texture are used for the icon? give back the answer in this format: <material1>, <material2>, <material3>.")["answer"]
    
    style = model.query(image, "is the Style of the icon: Fantasy, Detailed, Cute, 3D or realistic?")["answer"]
    
    details = model.caption(image, length="short")["caption"]
    
    description = model.caption(image, length="normal")["caption"]
    

    prompt = f"<subject: {main_subject.strip()}>, <style: {style.strip().lower()}>, <colors: {colors.strip().lower()}>, <materials: {materials.strip().lower()}>, <theme: {theme.strip().lower()} UI icon>, <details: {details.strip().lower()}>, <description: {description.strip().lower()}>"
    
    return prompt

def process_folder(folder_path):
    results = {}
    
    # Get all PNG files in the folder
    image_files = list(Path(folder_path).glob('*.png'))
    
    for img_path in image_files:
        try:
            # Load and process image
            image = Image.open(img_path)
            
            # Get structured description
            description = get_structured_description(image)
            
            # Store result using filename as key
            results[img_path.name] = description
            
            # Save individual txt file
            txt_path = img_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(description)
            
            print(f"Processed: {img_path.name}")
            print(f"Description: {description}")
            print(f"Saved to: {txt_path.name}\n")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
    
    output_path = os.path.join(folder_path, 'image_descriptions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll results saved to: {output_path}")
    print(f"Individual .txt files saved alongside each image")

# Process the folder
# suffix = "outline"
folder_path = f"./DoodlePixV4/A_normal/"

process_folder(folder_path)