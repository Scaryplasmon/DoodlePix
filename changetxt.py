import os

def replace_in_files(folder_path):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # Read the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                print(f"\nProcessing {filename}:")
                print(f"Original content: {content.strip()}")
                
                # Replace the text and track changes
                if '<tags:' in content:
                    modified_content = content.replace('<tags:', '[normal], <tags:')
                    print("Added [normal] tag")
                else:
                    modified_content = content
                    print("No <tags: found to add [normal] tag")
                    
                if 'background,' in modified_content:
                    modified_content = modified_content.replace('background,', 'background.')
                    print("Replaced background comma with period")
                else:
                    print("No background comma found to replace")

                if modified_content != content:
                    # Write back to the file only if changes were made
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(modified_content)
                    print(f"Modified content: {modified_content.strip()}")
                else:
                    print("No changes were needed for this file")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Use the script
folder_path = "DoodlePixV4/DoodlePixV5/edited_image/edit_prompt/"  # Replace with your folder path
# s = ["fantasy", "whimsical", "steampunk", "sci-fi"]
replace_in_files(folder_path)




#what to change
#remove UI icon>, from all the prompts
#
