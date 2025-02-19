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
                if ', <tags:' in content:
                    modified_content = content.replace(', <tags:', ',')
                    print("Removed , <tags:")
                else:
                    modified_content = content
                    print("No , <tags: found to remove")

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
folder_path = "DoodlePixV5_WIP/edit_prompt/"  # Replace with your folder path
# s = ["fantasy", "whimsical", "steampunk", "sci-fi"]
replace_in_files(folder_path)




#what to change
#remove UI icon>, from all the prompts
#
