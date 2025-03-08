import os
import re
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import numpy as np
from tqdm import tqdm

def parse_text_file(file_path):
    """Parse a single text file and extract structured data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Extract components using regex
    f_value_match = re.search(r'f(\d+),', content)
    style_match = re.search(r',\s*(\[\w+\])?,', content)
    
    # Extract f value (0-9)
    f_value = int(f_value_match.group(1)) if f_value_match else None
    
    # Extract style ([3D], [outline], [flat] or None)
    if style_match and style_match.group(1):
        style = style_match.group(1)
    else:
        style = "None"
    
    # Extract objects (everything between style and the first period)
    first_part = content.split('.')[0]
    # Remove the f-value and style part
    if style != "None":
        objects_part = first_part.split(style + ',')[1].strip()
    else:
        objects_part = first_part.split(',', 2)[2].strip() if len(first_part.split(',')) > 2 else ""
    
    objects = [obj.strip() for obj in objects_part.split(',')]
    
    # Extract colors (everything after the first period)
    if '.' in content:
        colors_part = content.split('.', 1)[1].strip()
        if colors_part:
            # Remove the final period if it exists
            if colors_part.endswith('.'):
                colors_part = colors_part[:-1]
            colors = [color.strip() for color in colors_part.split(',')]
            
            # Extract background color (the last one with "background")
            background_color = None
            for i in range(len(colors) - 1, -1, -1):
                if "background" in colors[i]:
                    background_color = colors[i]
                    break
            
            # Other colors (non-background)
            other_colors = [c for c in colors if "background" not in c]
        else:
            colors = []
            background_color = None
            other_colors = []
    else:
        colors = []
        background_color = None
        other_colors = []
    
    return {
        "f_value": f_value,
        "style": style,
        "objects": objects,
        "all_colors": colors,
        "background_color": background_color,
        "other_colors": other_colors,
        "raw_text": content
    }

def analyze_dataset(folder_path):
    """Analyze all text files in the given folder."""
    data = []
    
    # Get all text files
    text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    print(f"Found {len(text_files)} text files. Parsing...")
    
    # Parse each file
    for file_name in tqdm(text_files):
        file_path = os.path.join(folder_path, file_name)
        parsed_data = parse_text_file(file_path)
        parsed_data["file_name"] = file_name
        data.append(parsed_data)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    return df

def visualize_dataset(df):
    """Create visualizations for the dataset."""
    # Set the style
    sns.set(style="whitegrid")
    
    # Set the backend to 'Agg' which doesn't require a GUI
    import matplotlib
    matplotlib.use('Agg')
    
    plt.figure(figsize=(20, 15))
    
    # 1. Distribution of f values
    plt.subplot(3, 3, 1)
    sns.countplot(x='f_value', data=df, palette='viridis')
    plt.title('Distribution of f Values')
    plt.xlabel('f Value')
    plt.ylabel('Count')
    
    # 2. Distribution of styles
    plt.subplot(3, 3, 2)
    style_counts = df['style'].value_counts()
    sns.barplot(x=style_counts.index, y=style_counts.values, palette='viridis')
    plt.title('Distribution of Styles')
    plt.xlabel('Style')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 3. Top 10 objects
    plt.subplot(3, 3, 3)
    all_objects = [obj for sublist in df['objects'].tolist() for obj in sublist]
    object_counts = Counter(all_objects).most_common(10)
    sns.barplot(x=[item[0] for item in object_counts], y=[item[1] for item in object_counts], palette='viridis')
    plt.title('Top 10 Objects')
    plt.xlabel('Object')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # 4. Distribution of number of objects per file
    plt.subplot(3, 3, 4)
    object_counts = df['objects'].apply(len)
    sns.histplot(object_counts, kde=True, bins=max(10, object_counts.max()), palette='viridis')
    plt.title('Number of Objects per File')
    plt.xlabel('Number of Objects')
    plt.ylabel('Count')
    
    # 5. Top 10 background colors
    plt.subplot(3, 3, 5)
    bg_colors = [bg for bg in df['background_color'].dropna()]
    bg_color_counts = Counter(bg_colors).most_common(10)
    sns.barplot(x=[item[0] for item in bg_color_counts], y=[item[1] for item in bg_color_counts], palette='viridis')
    plt.title('Top 10 Background Colors')
    plt.xlabel('Background Color')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # 6. Top 10 non-background colors
    plt.subplot(3, 3, 6)
    all_colors = [color for sublist in df['other_colors'].tolist() for color in sublist if pd.notna(color)]
    color_counts = Counter(all_colors).most_common(10)
    sns.barplot(x=[item[0] for item in color_counts], y=[item[1] for item in color_counts], palette='viridis')
    plt.title('Top 10 Non-Background Colors')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # 7. Correlation between f value and number of objects
    plt.subplot(3, 3, 7)
    sns.boxplot(x='f_value', y=df['objects'].apply(len), data=df, palette='viridis')
    plt.title('Number of Objects by f Value')
    plt.xlabel('f Value')
    plt.ylabel('Number of Objects')
    
    # 8. Word cloud of all objects
    plt.subplot(3, 3, 8)
    all_objects_text = ' '.join(all_objects)
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_objects_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Objects')
    
    # 9. Word cloud of all colors
    plt.subplot(3, 3, 9)
    all_colors_text = ' '.join([color for sublist in df['all_colors'].tolist() for color in sublist if pd.notna(color)])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_colors_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Colors')
    
    plt.tight_layout()
    plt.savefig('dataset_overview.png', dpi=300)
    plt.close()
    
    # Additional visualizations
    
    # 1. Style distribution by f value
    plt.figure(figsize=(12, 8))
    style_f_counts = pd.crosstab(df['style'], df['f_value'])
    style_f_counts.plot(kind='bar', stacked=True)
    plt.title('Style Distribution by f Value')
    plt.xlabel('Style')
    plt.ylabel('Count')
    plt.legend(title='f Value')
    plt.tight_layout()
    plt.savefig('style_by_f_value.png', dpi=300)
    plt.close()
    
    # 2. Heatmap of object co-occurrence
    plt.figure(figsize=(15, 15))
    # Get top 20 objects
    top_objects = [item[0] for item in Counter(all_objects).most_common(20)]
    
    # Create co-occurrence matrix
    cooccurrence = np.zeros((len(top_objects), len(top_objects)))
    
    for obj_list in df['objects']:
        for i, obj1 in enumerate(top_objects):
            if obj1 in obj_list:
                for j, obj2 in enumerate(top_objects):
                    if obj2 in obj_list:
                        cooccurrence[i, j] += 1
    
    # Plot heatmap
    sns.heatmap(cooccurrence, annot=True, fmt='g', xticklabels=top_objects, yticklabels=top_objects)
    plt.title('Object Co-occurrence Matrix (Top 20 Objects)')
    plt.tight_layout()
    plt.savefig('object_cooccurrence.png', dpi=300)
    plt.close()
    
    # 3. Distribution of text length
    plt.figure(figsize=(10, 6))
    text_lengths = df['raw_text'].apply(len)
    sns.histplot(text_lengths, kde=True, bins=30)
    plt.title('Distribution of Text Length')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('text_length_distribution.png', dpi=300)
    plt.close()
    
    print("Visualizations saved as PNG files.")
    
    # Return some basic statistics
    stats = {
        "total_files": len(df),
        "unique_f_values": df['f_value'].nunique(),
        "unique_styles": df['style'].nunique(),
        "unique_objects": len(set(all_objects)),
        "unique_colors": len(set([color for sublist in df['all_colors'].tolist() for color in sublist if pd.notna(color)])),
        "avg_objects_per_file": df['objects'].apply(len).mean(),
        "avg_colors_per_file": df['all_colors'].apply(len).mean(),
    }
    
    return stats

def main():
    folder_path = "data/DoodlePixV6/edit_prompt/"
    
    if not os.path.exists(folder_path):
        print(f"Error: The path '{folder_path}' does not exist.")
        return
    
    df = analyze_dataset(folder_path)
    
    # Save the parsed data to CSV for further analysis
    df.to_csv('parsed_dataset.csv', index=False)
    print("Parsed data saved to 'parsed_dataset.csv'")
    
    # Create visualizations
    stats = visualize_dataset(df)
    
    # Print statistics
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\nVisualization complete! Check the current directory for the generated images.")

if __name__ == "__main__":
    main()