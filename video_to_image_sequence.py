import cv2
import pandas as pd
import json
import os
import re

# Define paths
base_video_path = "D:/thesis dataset/dataset_mosxos/debate_recordings"  # Change this to your video file path
base_annotation_path = "D:/thesis dataset/dataset_mosxos/emotion_annotations/aggregated_external_annotations/"  # Change this to your CSV file path
output_base_path = "D:/thesis dataset/dataset_mosxos/output/"  # Base directory to save output images and JSON files

# Create base output directory if it doesn't exist
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# Function to extract base name from video file
def get_base_name(filename):
    match = re.match(r'(p\d+)_\d+', filename.lower())  # Normalize to lowercase
    if match:
        return match.group(1)
    return None

# List video files in the base video path
video_files = [f for f in os.listdir(base_video_path) if f.endswith('.mp4') and re.match(r'p\d+_\d+\.mp4', f.lower())]

# Process each video file
for video_file in video_files:
    print(f"Processing video file: {video_file}")
    base_name = get_base_name(video_file)
    if base_name is None:
        print(f"Skipping video file {video_file} due to invalid base name.")
        continue

    video_path = os.path.join(base_video_path, video_file)
    csv_path = os.path.join(base_annotation_path, f"{base_name}.external.csv")

    # Check if the corresponding CSV file exists
    if not os.path.exists(csv_path):
        print(f"Annotation file for {video_file} not found. Skipping.")
        continue

    # Load annotations
    annotations = pd.read_csv(csv_path)

    # Create output directory for this video
    video_output_path = os.path.join(output_base_path, base_name)
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    frame_interval = int(fps * 0.1)  # Calculate interval for 0.1 seconds

    # Initialize variables
    frame_count = 0
    annotations_list = []

    # Process the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is at the 0.1 second interval
        if frame_count % frame_interval == 0:
            timestamp = (frame_count) / fps
            if timestamp>0:
               annotation_index = int((timestamp-0.1) // 5)  # Shift timestamp by 0.1 seconds and then calculate index
            else:
               annotation_index = int((timestamp) // 5)  # Shift timestamp by 0.1 seconds and then calculate index

            if annotation_index < len(annotations):
                valence = int(annotations.iloc[annotation_index]['valence'])  # Convert to int
                arousal = int(annotations.iloc[annotation_index]['arousal'])  # Convert to int

                # Print or log the values retrieved
                print(f"Frame {frame_count}: Valence = {valence}, Arousal = {arousal}")

                # Save frame as image directly in video output path
                image_filename = f"{base_name}_frame_{frame_count:06d}.jpg"  # Zero-padded for sorted order
                image_path = os.path.join(video_output_path, image_filename)
                cv2.imwrite(image_path, frame)

                # Append annotation to list
                annotations_list.append({
                    "name": image_filename,
                    "timestamp": timestamp,
                    "valence": valence,
                    "arousal": arousal
                })

        frame_count += 1

    # Release video capture
    cap.release()

    # Save annotations to JSON file directly in video output path
    output_json_path = os.path.join(video_output_path, f"{base_name}_annotations.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(annotations_list, json_file, indent=4)

    print(f"Finished processing {video_file}. Frames and annotations saved to {video_output_path}.")
