import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter


class CustomImageSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=32):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequence_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if
                                 os.path.isdir(os.path.join(root_dir, d))]
        self.label_map = self.create_label_map()
        self.check_labels()

    def create_label_map(self):
        label_set = set()
        for folder in self.sequence_folders:
            json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
            for json_file in json_files:
                annotation_file = os.path.join(folder, json_file)
                try:
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                        for ann in annotations:
                            label_set.add(ann['label'])
                except json.JSONDecodeError as e:
                    print(f"  Error decoding JSON file {annotation_file}: {e}")
                    continue
        label_map = {label: idx for idx, label in enumerate(sorted(label_set))}
        return label_map

    def check_labels(self):
        valid_labels = set(self.label_map.keys())
        for folder in self.sequence_folders:
            json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
            for json_file in json_files:
                annotation_file = os.path.join(folder, json_file)
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                    for ann in annotations:
                        label = ann['label']
                        if label not in valid_labels:
                            raise ValueError(f"Invalid label '{label}' found in {annotation_file}")

    def __len__(self):
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        folder = self.sequence_folders[idx]
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        if not json_files:
            raise ValueError(f"No JSON files found in folder {folder}")

        annotation_file = os.path.join(folder, json_files[0])

        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        images = []
        labels = []
        image_paths = []

        # Collect images and labels
        for ann in annotations:
            img_path = os.path.join(folder, ann['name'])
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image file {img_path} does not exist")
            image_paths.append(img_path)
            labels.append(self.label_map[ann['label']])

        # Sort image paths based on the filename
        image_paths, labels = zip(*sorted(zip(image_paths, labels), key=lambda x: x[0]))

        # Load and transform images after sorting
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        if not images or not labels:
            raise ValueError(f"No images or labels found for index {idx}")

        images = torch.stack(images)
        labels = torch.tensor(labels)

        # Handle padding or truncation
        if len(images) < self.sequence_length:
            num_to_pad = self.sequence_length - len(images)
            last_frame = images[-1]
            last_label = labels[-1]
            padding = last_frame.unsqueeze(0).repeat(num_to_pad, 1, 1, 1)
            images = torch.cat((images, padding), dim=0)
            label_padding = torch.full((num_to_pad,), last_label, dtype=torch.long)
            labels = torch.cat((labels, label_padding), dim=0)
        else:
            images = images[:self.sequence_length]
            labels = labels[:self.sequence_length]

        # Compute the most frequent label for the sequence (video)
        label_counts = Counter(labels.tolist())
        most_frequent_label = label_counts.most_common(1)[0][0]

        images = images.permute(1, 0, 2, 3)  # Change from (T, C, H, W) to (C, T, H, W)

        return images, torch.tensor(most_frequent_label)

    def get_label_map(self):
        """Return the label map for inspection."""
        return self.label_map

    def count_label_instances(self):
        """Count the number of instances for each label based on the most frequent label in the first 32-frame sequence of each folder."""
        label_counts = Counter()

        for folder in self.sequence_folders:
            json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
            if not json_files:
                continue

            annotation_file = os.path.join(folder, json_files[0])

            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                labels = [self.label_map[ann['label']] for ann in annotations]

                # Only process the first 32 frames
                if len(labels) > self.sequence_length:
                    labels = labels[:self.sequence_length]

                # Compute the most frequent label in this 32-frame sequence
                sequence_label_counts = Counter(labels)
                most_frequent_label = sequence_label_counts.most_common(1)[0][0]

                # Count the most frequent label for the sequence
                label_counts[most_frequent_label] += 1

        return label_counts


# Define normalization transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Create datasets
training_dataset = CustomImageSequenceDataset(root_dir='D:\\thesis dataset\\split_dataset\\training_clear',
                                              transform=transform)
validation_dataset = CustomImageSequenceDataset(root_dir='D:\\thesis dataset\\split_dataset\\validate_clear',
                                                transform=transform)

# Print label map for both training and validation datasets
print("Training Dataset Label Map:")
for label, idx in training_dataset.get_label_map().items():
    print(f"{label}: {idx}")

print("\nValidation Dataset Label Map:")
for label, idx in validation_dataset.get_label_map().items():
    print(f"{label}: {idx}")

# Count and print label instances
print("\nTraining Dataset Label Counts Based on Sequences:")
train_label_counts = training_dataset.count_label_instances()
for label, count in train_label_counts.items():
    print(f"{label}: {count}")

print("\nValidation Dataset Label Counts Based on Sequences:")
val_label_counts = validation_dataset.count_label_instances()
for label, count in val_label_counts.items():
    print(f"{label}: {count}")

# Create DataLoaders
train_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
