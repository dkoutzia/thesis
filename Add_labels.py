import os
import json

# Define the directory containing the normalized JSON files
normalized_dir = r'D:\thesis dataset\split_dataset\training_clear'

# Function to determine the label based on valence and arousal
def determine_label():

        return "high negative emotion"


# Traverse through all subfolders and locate the normalized JSON files
for root, dirs, files in os.walk(normalized_dir):
    for file in files:
        if file.endswith('_normalized.json'):
            json_path = os.path.join(root, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"Processing file: {json_path}")

                # Iterate over frames to add labels
                for frame_data in data:
                    valence = frame_data['valence']
                    arousal = frame_data['arousal']
                    frame_data['label'] = determine_label()

            # Save the JSON file with labels
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)

print("Labels added and saved to normalized JSON files.")
