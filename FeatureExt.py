# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:51:07 2025

@author: zbukh
"""

import cv2
import os

# Define paths
input_dir = r"C:\Users\zbukh\OneDrive\Desktop\HAr\Human Activity Recognition - Video Dataset"  # Update with the path to your dataset folder
output_dir = r"C:\Users\zbukh\OneDrive\Desktop\HAr\extracted frames"  # Update with the path where you want to save frames

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def extract_frames(video_path, output_folder, activity_name):
    # Read video
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when no frames are left
        
        # Define the frame filename
        frame_filename = os.path.join(output_folder, f'{activity_name}_frame_{frame_count}.jpg')
        
        # Save frame as an image
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f'Frames extracted for video: {video_path}')

# Loop through each activity folder and extract frames from each video
for activity in os.listdir(input_dir):
    activity_path = os.path.join(input_dir, activity)
    
    if os.path.isdir(activity_path):  # Ensure it's a folder
        for video_file in os.listdir(activity_path):
            if video_file.endswith('.mp4'):  # Process only .mp4 videos
                video_path = os.path.join(activity_path, video_file)
                activity_output_folder = os.path.join(output_dir, activity)
                os.makedirs(activity_output_folder, exist_ok=True)  # Create a folder for each activity
                
                # Extract frames for each video
                extract_frames(video_path, activity_output_folder, activity)

print("Frame extraction completed.")
