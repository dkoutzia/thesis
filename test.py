import os
import torch
from torchvision import transforms
from PIL import Image
from dataloader import training_dataset
from torchvision.models.video import r3d_18
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FineTunedResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet3D, self).__init__()
        self.base_model = r3d_18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Define the path to the test sequence folder and best model weights
test_sequence_folder = 'D:\\thesis dataset\\split_dataset\\test\\processed1'
best_model_weights = 'top_model_weights.pth'

# Define transformation (same as the training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the model and set it to evaluation mode
num_classes = len(training_dataset.label_map)
model = FineTunedResNet3D(num_classes=num_classes)

# Load the best weights
model.load_state_dict(torch.load(best_model_weights))

# Set the model to evaluation mode (important for inference)
model.eval()

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to load and preprocess images from a folder
def load_image_sequence(sequence_folder, transform, sequence_length=32):
    image_paths = [os.path.join(sequence_folder, f) for f in os.listdir(sequence_folder) if f.endswith('.png')]
    image_paths = sorted(image_paths)  # Sort the image paths to maintain order

    images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        images.append(image)

    # Stack images into a tensor of shape (sequence_length, C, H, W)
    images = torch.stack(images)

    # Handle padding or truncation
    if len(images) < sequence_length:
        num_to_pad = sequence_length - len(images)
        last_frame = images[-1]
        padding = last_frame.unsqueeze(0).repeat(num_to_pad, 1, 1, 1)
        images = torch.cat((images, padding), dim=0)
    else:
        images = images[:sequence_length]

    # Change the shape to (C, T, H, W) for the model
    images = images.permute(1, 0, 2, 3)
    return images

# Load the test image sequence
test_sequence = load_image_sequence(test_sequence_folder, transform)
test_sequence = test_sequence.unsqueeze(0)  # Add batch dimension

# Move the sequence to GPU if available
test_sequence = test_sequence.to(device)

# Perform prediction
with torch.no_grad():  # Ensure no gradients are calculated
    output = model(test_sequence)

    # Apply softmax to get probabilities (confidence scores)
    probabilities = F.softmax(output, dim=1)

    # Get the predicted class label
    _, predicted_label = torch.max(probabilities, 1)

# Print the predicted label and confidence scores for each class
print(f'Predicted label for the test sequence: {predicted_label.item()}')
print(f'Confidence scores for each class: {probabilities.cpu().numpy()}')
