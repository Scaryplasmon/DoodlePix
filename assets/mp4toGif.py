import os
from moviepy.editor import VideoFileClip

# Set the directory you want to process; here we use the current folder
folder = "."

# Loop through all files in the directory
for filename in os.listdir(folder):
    if filename.lower().endswith(".mp4"):
        mp4_path = os.path.join(folder, filename)
        gif_filename = filename[:-4] + ".gif"
        gif_path = os.path.join(folder, gif_filename)
        print(f"Converting {filename} to {gif_filename}...")
        try:
            # Load the video and set the target fps
            clip = VideoFileClip(mp4_path)
            # First resample the video to the desired fps
            clip = clip.set_fps(2)
            # Then write the gif with the same fps to maintain timing
            clip.write_gif(gif_path, fps=2, program='ffmpeg')
            clip.close()
            print(f"Successfully created {gif_filename}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
