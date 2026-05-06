import os
import shutil
import cv2
import json
import numpy as np
import random
from albumentations import Rotate, Compose
from albumentations.pytorch import ToTensorV2

dataset_dir = r'D:\thesis dataset\split_dataset\training_clear'
output_dir = r'D:\thesis dataset\split_dataset\training_clear'

# List all folders in the dataset directory
folders = [folder for folder in os.listdir(dataset_dir)
           if os.path.isdir(os.path.join(dataset_dir, folder))]

# Select 200 random folders
num_folders_to_process = 200
if len(folders) > num_folders_to_process:
    selected_folders = random.sample(folders, num_folders_to_process)
else:
    selected_folders = folders

# Define augmentation pipeline: random rotation (max 10 degrees)
augmentation = Compose([
    Rotate(limit=4, p=1.0),  # Always apply rotation
    ToTensorV2(),
])

def augment_images_in_folder(folder_path):
    """
    Apply random rotation to all images in the folder and save augmented images.
    Copy all JSON files from the original folder to the augmented folder.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                if image is None:
                    continue

                # Apply augmentation
                augmented = augmentation(image=image)
                augmented_image = augmented['image']

                # Convert augmented image to numpy array
                if isinstance(augmented_image, np.ndarray):
                    augmented_image_np = augmented_image
                else:
                    augmented_image_np = augmented_image.permute(1, 2, 0).numpy()

                # Convert image to uint8 if it's in the range [0, 1]
                if augmented_image_np.max() <= 1.0:
                    augmented_image_np = (augmented_image_np * 255).astype(np.uint8)

                # Construct output path for augmented image
                output_folder_name = f"{os.path.basename(root)}_rotated"
                output_folder_path = os.path.join(output_dir, output_folder_name)
                os.makedirs(output_folder_path, exist_ok=True)
                augmented_image_name = file
                augmented_image_path = os.path.join(output_folder_path, augmented_image_name)

                # Save augmented image
                cv2.imwrite(augmented_image_path, augmented_image_np)

        # Copy all JSON files from the original folder to the augmented folder
        json_files = [f for f in os.listdir(root) if f.endswith('.json')]
        for json_file in json_files:
            shutil.copy(os.path.join(root, json_file), output_folder_path)

# Process only the selected folders for augmentation
for folder_name in selected_folders:
    folder_path = os.path.join(dataset_dir, folder_name)
    augment_images_in_folder(folder_path)

print("Augmentation complete.")
