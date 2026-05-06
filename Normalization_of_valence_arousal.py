import os
import json

# Define the directory containing the dataset and the directory to save the normalized data
base_dir = r'D:\thesis dataset\i bug\output'



# Initialize lists to hold all valence and arousal values
valences = []
arousals = []

# Step 1: Traverse through all subfolders and locate the JSON files
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(root, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                for entry in data:
                    valences.append(entry['valence'])
                    arousals.append(entry['arousal'])

# Step 2: Determine the maximum and minimum valence and arousal values
max_valence = 10
min_valence = -10
max_arousal = 10
min_arousal = -10
print(max_valence,min_valence,min_arousal,max_arousal)

# Step 3: Normalize the valence and arousal values to the range [-1, 1]
def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0  # Avoid division by zero if all values are the same
    return 2 * (value - min_value) / (max_value - min_value) - 1

# Step 4: Traverse through all subfolders and locate the JSON files again to update them
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.json'):
            json_path = os.path.join(root, file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                for entry in data:
                    entry['valence'] = normalize(entry['valence'], min_valence, max_valence)
                    entry['arousal'] = normalize(entry['arousal'], min_arousal, max_arousal)

            # Create corresponding path in the normalized directory
            relative_path = os.path.relpath(root, base_dir)
            normalized_subdir = os.path.join(base_dir, relative_path)
            if not os.path.exists(normalized_subdir):
                os.makedirs(normalized_subdir)

            # Save the normalized JSON file with "_normalized" suffix
            normalized_json_path = os.path.join(normalized_subdir, file.replace('.json', '_normalized.json'))
            with open(normalized_json_path, 'w') as f:
                json.dump(data, f, indent=4)

print("Normalization complete and saved to new location.")
