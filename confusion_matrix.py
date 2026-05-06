import torch
import torch.nn as nn
from dataloader import val_dataloader
from torchvision.models.video import r3d_18
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from dataloader import validation_dataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Define the model architecture (similar to your training script)
class FineTunedResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet3D, self).__init__()
        self.base_model = r3d_18(pretrained=False)  # Pretrained is set to False, as we will load the fine-tuned weights
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


# Load the saved model weights
num_classes = len(validation_dataset.label_map)  # Assuming the same label map is used
model = FineTunedResNet3D(num_classes=num_classes)
model.load_state_dict(torch.load('top_model_weights.pth'))  # Load the best model weights

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Put the model in evaluation mode
model.eval()

# Initialize variables for metrics
all_true_labels = []
all_predicted_labels = []

# Define loss function (if you want to compute the loss)
criterion = torch.nn.CrossEntropyLoss()

# Evaluate the model
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    running_val_loss = 0

    for val_images, val_labels in val_dataloader:
        val_images, val_labels = val_images.to(device), val_labels.to(device)

        # Forward pass
        val_outputs = model(val_images)
        val_loss = criterion(val_outputs, val_labels)
        running_val_loss += val_loss.item()

        _, predicted_val = torch.max(val_outputs, 1)

        # Track true and predicted labels for confusion matrix
        all_true_labels.extend(val_labels.cpu().numpy())
        all_predicted_labels.extend(predicted_val.cpu().numpy())

        # Calculate accuracy
        total_correct += (predicted_val == val_labels).sum().item()
        total_samples += val_labels.size(0)

    val_accuracy = 100 * total_correct / total_samples
    val_loss = running_val_loss / len(val_dataloader)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
class_names = [str(i) for i in range(num_classes)]
plot_confusion_matrix(conf_matrix, classes=class_names, title='Confusion Matrix (Validation Set)')
