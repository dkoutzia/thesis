import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import training_dataset, train_dataloader, validation_dataset, val_dataloader
from torchvision.models.video import r3d_18
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FineTunedResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet3D, self).__init__()
        self.base_model = r3d_18(pretrained=True)
        # Example:
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)  # First FC layer
    def forward(self, x):
        return self.base_model(x)

# Plot function
def plot_metrics(x, y, ylabel, title, label=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Initialize model
num_classes = len(training_dataset.label_map)
print(num_classes)
model = FineTunedResNet3D(num_classes=num_classes)
print(model)

# Freeze all parameters except for the fully connected layer and layer4
for name, param in model.named_parameters():
    if "fc" in name or "layer4" in name:
        param.requires_grad = True
    else:
        param.requires_grad=False

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.00008,
    weight_decay=1e-04
)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 15
epochs = list(range(1, num_epochs + 1))

best_val_accuracy = 0
train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []
label_precisions = {label: [] for label in range(num_classes)}
label_recalls = {label: [] for label in range(num_classes)}
label_f1_scores = {label: [] for label in range(num_classes)}

print("Starting training loop...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batch_counter = 0

    for images, labels in train_dataloader:
        batch_counter += 1
        print("Processing batch ", batch_counter)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = 100 * correct_predictions / total_samples
    train_accuracies.append(train_accuracy)
    train_losses.append(running_loss / len(train_dataloader))

    model.eval()
    total_correct_val = 0
    total_samples_val = 0
    val_running_loss = 0

    # Initialize precision, recall, F1, and confusion matrix trackers
    true_positives = {label: 0 for label in range(num_classes)}
    false_positives = {label: 0 for label in range(num_classes)}
    false_negatives = {label: 0 for label in range(num_classes)}
    label_total = {label: 0 for label in range(num_classes)}

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for val_images, val_labels in val_dataloader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()
            _, predicted_val = torch.max(val_outputs, 1)
            total_correct_val += (predicted_val == val_labels).sum().item()
            total_samples_val += val_labels.size(0)

            # Append to confusion matrix trackers
            all_true_labels.extend(val_labels.cpu().numpy())
            all_predicted_labels.extend(predicted_val.cpu().numpy())

            # Update precision, recall, and F1 metrics
            for label, pred in zip(val_labels, predicted_val):
                label_total[label.item()] += 1
                if label == pred:
                    true_positives[label.item()] += 1
                else:
                    false_positives[pred.item()] += 1
                    false_negatives[label.item()] += 1

    val_accuracy = 100 * total_correct_val / total_samples_val
    val_accuracies.append(val_accuracy)
    val_losses.append(val_running_loss / len(val_dataloader))

    # Calculate precision, recall, and F1 for each label
    for label in range(num_classes):
        if true_positives[label] + false_positives[label] > 0:
            precision = 100 * true_positives[label] / (true_positives[label] + false_positives[label])
        else:
            precision = 0.0
        if true_positives[label] + false_negatives[label] > 0:
            recall = 100 * true_positives[label] / (true_positives[label] + false_negatives[label])
        else:
            recall = 0.0

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        label_precisions[label].append(precision)
        label_recalls[label].append(recall)
        label_f1_scores[label].append(f1_score)

        print(f'Label {label}: Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1_score:.2f}%')

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {running_loss / len(train_dataloader):.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_running_loss / len(val_dataloader):.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Save the best model weights
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'top_model_weights.pth')
        print("Best model weights saved.")

torch.save(model.state_dict(), 'last_model_weights.pth')
print("Last model weights saved.")
print('Finished Training')

# Plot train accuracy and loss
plot_metrics(epochs, train_accuracies, 'Accuracy', 'Training Accuracy')
plot_metrics(epochs, train_losses, 'Loss', 'Training Loss')

# Plot validation accuracy and loss
plot_metrics(epochs, val_accuracies, 'Accuracy', 'Validation Accuracy')
plot_metrics(epochs, val_losses, 'Loss', 'Validation Loss')

# Plot precision, recall, and F1 score for each label
for label in range(num_classes):
    plot_metrics(epochs, label_precisions[label], 'Precision (%)', f'Precision for Label {label}')
    plot_metrics(epochs, label_recalls[label], 'Recall (%)', f'Recall for Label {label}')
    plot_metrics(epochs, label_f1_scores[label], 'F1 Score', f'F1 Score for Label {label}')

# Calculate and plot confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
class_names = [str(i) for i in range(num_classes)]
plot_confusion_matrix(conf_matrix, classes=class_names, title='Confusion Matrix')
