import os
import random
import cv2
import numpy as np
import shutil


dataset_dir = r'D:\thesis dataset\split_dataset\training'
output_dir = r'D:\thesis dataset\split_dataset\training'

# List all folders in the dataset directory, excluding those starting with 'p'
folders = [folder for folder in os.listdir(dataset_dir)
           if os.path.isdir(os.path.join(dataset_dir, folder)) and folder.isdigit()]

# Calculate how many folders to select (about one sixtieth)
num_folders_to_augment = len(folders) // 10

# Randomly select folders to augment
folders_to_rotate = random.sample(folders, num_folders_to_augment)


# Custom function to apply rotation with padding
def apply_rotation(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Find bounding box of the non-black area
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = rotated_image[y:y + h, x:x + w]

    return cropped_image


# Custom function to apply brightness and contrast adjustment
def apply_brightness_contrast(image, brightness, contrast):
    # Apply contrast
    f = 131 * (contrast + 127) / (127 * (131 - contrast))
    alpha_c = f
    gamma_c = 127 * (1 - f)

    # Apply brightness
    new_image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c + brightness)
    return new_image


# Function to process and save augmented images
def augment_and_save_images(folder, augment_function, output_folder):
    folder_path = os.path.join(dataset_dir, folder)
    output_folder_name = folder + "10"
    output_folder_path = os.path.join(output_folder, output_folder_name)

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image = cv2.imread(image_path)
            augmented_image = augment_function(image)
            output_image_path = os.path.join(output_folder_path, image_name)
            cv2.imwrite(output_image_path, augmented_image)
            json_files = [json_file for json_file in os.listdir(folder_path) if json_file.endswith('.json')]
            for json_file in json_files:
                shutil.copy(os.path.join(folder_path, json_file), output_folder_path)

# Apply rotation to the selected folders
for folder in folders_to_rotate:
    augment_and_save_images(folder, lambda img: apply_rotation(img, angle=10), output_dir)

