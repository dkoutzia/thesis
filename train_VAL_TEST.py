import os
import shutil
import random

def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' does not exist.")
        return

    # Create output directories if they don't exist
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validate')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all the subdirectories in the dataset directory
    all_folders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not all_folders:
        print(f"No folders found in dataset directory '{dataset_dir}'.")
        return

    print(f"Found {len(all_folders)} folders in the dataset directory.")

    # Shuffle the folders randomly
    random.shuffle(all_folders)

    # Calculate the number of folders for each split
    total_folders = len(all_folders)
    train_count = int(total_folders * train_ratio)
    val_count = int(total_folders * val_ratio)
    test_count = total_folders - train_count - val_count  # Ensure all folders are used

    # Split the folders
    train_folders = all_folders[:train_count]
    val_folders = all_folders[train_count:train_count + val_count]
    test_folders = all_folders[train_count + val_count:]

    # Function to copy folders to their respective directories
    def copy_folders(folders, destination):
        for folder in folders:
            src_path = os.path.join(dataset_dir, folder)
            dest_path = os.path.join(destination, folder)
            if not os.path.exists(src_path):
                print(f"Source path '{src_path}' does not exist.")
                continue
            print(f"Copying from '{src_path}' to '{dest_path}'")
            shutil.copytree(src_path, dest_path)

    print(f"Copying {len(train_folders)} folders to train directory...")
    copy_folders(train_folders, train_dir)

    print(f"Copying {len(val_folders)} folders to validate directory...")
    copy_folders(val_folders, val_dir)

    print(f"Copying {len(test_folders)} folders to test directory...")
    copy_folders(test_folders, test_dir)

    print(f"Total folders: {total_folders}")
    print(f"Training folders: {len(train_folders)}")
    print(f"Validation folders: {len(val_folders)}")
    print(f"Test folders: {len(test_folders)}")

# Example usage
dataset_dir = r'D:\thesis dataset\my_dataset'
output_dir = r'D:\thesis dataset\split_dataset'
split_dataset(dataset_dir, output_dir)
