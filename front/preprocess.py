# preprocess.py

from moviepy.editor import VideoFileClip
import os

def preprocess(video_file):
    try:
        # Load the video file
        video_clip = VideoFileClip(video_file)
        
        # Cut the video to the first 5 seconds
        video_clip = video_clip.subclip(0, 5)
        
        # Save the processed video
        output_file = "processed_video.mp4"
        video_clip.write_videofile(output_file, codec='libx264')
        
        return True
    except Exception as e:
        print(f"Error processing video: {e}")
        return False