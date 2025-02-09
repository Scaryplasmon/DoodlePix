import os
from pathlib import Path
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import json

class CaptionAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.data = []
        
    def parse_caption_file(self, file_path):
        """Parse a single caption file and extract structured data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Updated regex patterns to match the current format
        patterns = {
            'detail_level': r'<k:(\d+)',  # Changed from <k:(\d+)/10>
            'style': r'<p:(\w+)>',
            'world': r'<w:(\w+)',  # Changed from <t:(\w+) UI
        }
        
        data = {'filename': file_path.stem}
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                if key == 'detail_level':
                    value = int(value)
                data[key] = value
            else:
                print(f"Warning: Could not find {key} in {file_path.name}")
                data[key] = None
                
        return data
    
    def analyze_folder(self):
        """Analyze all txt files in the folder"""
        txt_files = list(self.folder_path.glob('*.txt'))
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.folder_path}")
            
        for file_path in txt_files:
            try:
                data = self.parse_caption_file(file_path)
                self.data.append(data)
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
        
        df = pd.DataFrame(self.data)
        
        # Print data summary for debugging
        print("\nData Summary:")
        print(f"Total files processed: {len(df)}")
        print("\nColumns found:")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")
            
        return df
    
    def create_visualizations(self, output_dir='analysis_results'):
        """Create and save various visualizations"""
        df = self.analyze_folder()
        
        if df.empty:
            raise ValueError("No data to visualize!")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        try:
            # 1. Detail Level Distribution
            if df['detail_level'].notna().any():
                plt.figure(figsize=(12, 6))
                sns.histplot(data=df, x='detail_level', bins=range(0, 12))
                plt.title('Distribution of Detail Levels', fontsize=14)
                plt.xlabel('Detail Level (1-10)', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.savefig(f'{output_dir}/detail_distribution.png', bbox_inches='tight', dpi=300)
                plt.close()
            
            # 2. World Distribution
            if df['world'].notna().any():
                plt.figure(figsize=(10, 6))
                world_counts = df['world'].value_counts()
                sns.barplot(x=world_counts.index, y=world_counts.values)
                plt.title('Distribution of Worlds', fontsize=14)
                plt.xlabel('World Type', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45)
                plt.savefig(f'{output_dir}/world_distribution.png', bbox_inches='tight', dpi=300)
                plt.close()
            
            # 3. Style Distribution
            if df['style'].notna().any():
                plt.figure(figsize=(10, 6))
                style_counts = df['style'].value_counts()
                sns.barplot(x=style_counts.index, y=style_counts.values)
                plt.title('Distribution of Styles', fontsize=14)
                plt.xlabel('Style Type', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45)
                plt.savefig(f'{output_dir}/style_distribution.png', bbox_inches='tight', dpi=300)
                plt.close()
            
            # 4. Detail Level by World
            if df['detail_level'].notna().any() and df['world'].notna().any():
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=df, x='world', y='detail_level')
                plt.title('Detail Levels by World', fontsize=14)
                plt.xlabel('World Type', fontsize=12)
                plt.ylabel('Detail Level', fontsize=12)
                plt.xticks(rotation=45)
                plt.savefig(f'{output_dir}/detail_by_world.png', bbox_inches='tight', dpi=300)
                plt.close()
            
            # 5. Style vs World Heatmap
            if df['style'].notna().any() and df['world'].notna().any():
                plt.figure(figsize=(12, 8))
                style_world_counts = pd.crosstab(df['style'], df['world'])
                sns.heatmap(style_world_counts, annot=True, fmt='d', cmap='YlOrRd')
                plt.title('Style vs World Distribution', fontsize=14)
                plt.xlabel('World Type', fontsize=12)
                plt.ylabel('Style Type', fontsize=12)
                plt.savefig(f'{output_dir}/style_world_heatmap.png', bbox_inches='tight', dpi=300)
                plt.close()
            
            # Save summary statistics
            summary = {
                'total_images': len(df),
                'average_detail_level': float(df['detail_level'].mean()) if df['detail_level'].notna().any() else None,
                'world_distribution': df['world'].value_counts().to_dict() if df['world'].notna().any() else {},
                'style_distribution': df['style'].value_counts().to_dict() if df['style'].notna().any() else {}
            }
            
            with open(f'{output_dir}/summary_stats.json', 'w') as f:
                json.dump(summary, f, indent=4)
                
            print(f"\nVisualizations saved to: {output_dir}")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze caption files and create visualizations')
    parser.add_argument('--folder_path', '-i', type=str, required=True,
                      help='Path to folder containing caption txt files')
    parser.add_argument('--output_dir', '-o', type=str, default='analysis_results',
                      help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    try:
        analyzer = CaptionAnalyzer(args.folder_path)
        analyzer.create_visualizations(args.output_dir)
    except Exception as e:
        print(f"\nError: {str(e)}")