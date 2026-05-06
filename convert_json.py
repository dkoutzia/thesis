import os
import os
import json
import re


def convert_to_png(name):
    # Add .png extension if it's not already there
    if not name.lower().endswith('.png'):
        return f"{name}.png"
    return name


def update_name_in_json_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.json'):
                json_path = os.path.join(root, filename)
                print(f"Processing file: {json_path}")

                # Read the JSON file
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Update each object in the JSON array
                for obj in data:
                    if 'name' in obj:
                        obj['name'] = convert_to_png(obj['name'])

                # Write back to the JSON file
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                print(f"Updated file: {json_path}")


# Example usage
root_directory = r'D:\thesis dataset\i_bug\dataset'
update_name_in_json_files(root_directory)
