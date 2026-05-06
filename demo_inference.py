import cv2
import os
from pathlib import Path
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.video import r3d_18
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import training_dataset

class FineTunedResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet3D, self).__init__()
        self.base_model = r3d_18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Function to preprocess image ensuring it's in uint8 format
def preprocess_image(image):
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Ensure the image is in uint8 format
        if image_rgb.dtype != np.uint8:
            image_rgb = (255 * (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min())).astype(np.uint8)
        return image_rgb
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

label_map = {
    0: "High Negative Emotion",
    1: "High Positive Emotion",
    2: "Low/Neutral Emotion"
}

# Function to process video and extract frames at 10 FPS with bounding boxes and confidence scores
def process_video_with_bounding_boxes(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Get the original frame rate and video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up video writer for output video with original FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (frame_width, frame_height))

    print(f"Original Frame rate: {original_fps} FPS")
    frame_count = 0
    best_face_list = []  # Store face images for prediction
    frame_buffer = []  # Buffer to hold the last 96 frames for retroactive prediction application
    current_prediction = None  # Store the current prediction for continuous use

    # Initialize prediction counter
    prediction_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in every frame
        image_rgb = preprocess_image(frame)
        if image_rgb is None:
            continue

        detected_faces = DeepFace.extract_faces(
            img_path=image_rgb,
            enforce_detection=False,
            detector_backend='yolov8'
        )

        # Draw bounding boxes for detected faces
        if detected_faces is not None and len(detected_faces) > 0:
            for face_data in detected_faces:
                region = face_data['facial_area']  # Get face bounding box coordinates
                (x, y, w, h) = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if frame_count % 3 == 0:
                    face_image = face_data['face']
                    if face_image.dtype != np.uint8:
                        face_image = (255 * (face_image - face_image.min()) / (
                                    face_image.max() - face_image.min())).astype(np.uint8)

                    best_face_list.append(face_image)

                # Ensure we have exactly 32 frames for prediction
                if len(best_face_list) == 32:
                    # Convert the list of faces to tensor
                    face_sequence = [transform(Image.fromarray(face.astype(np.uint8))) for face in best_face_list]
                    face_sequence = torch.stack(face_sequence)
                    face_sequence = face_sequence.permute(1, 0, 2, 3).unsqueeze(0)  # Add batch dimension
                    face_sequence = face_sequence.to(device)

                    # Perform prediction
                    with torch.no_grad():
                        output = model(face_sequence)
                        probabilities = F.softmax(output, dim=1)

                        # Get the predicted label and confidence scores for all classes
                        predicted_label = torch.argmax(probabilities, dim=1).item()  # Get the predicted label
                        confidence_scores = probabilities.squeeze().cpu().numpy()  # Convert to numpy array

                        # Store the current prediction for retroactive application to previous 96-frame block
                        current_prediction = (predicted_label, confidence_scores.tolist())

                        # Now, apply the prediction to the previous block of 96 frames in the buffer
                        for buffered_frame in frame_buffer:
                            for idx, conf_score in enumerate(confidence_scores):
                                label_text = f"Class: {label_map[idx]}, Conf: {conf_score:.2f}"
                                cv2.putText(buffered_frame, label_text, (x, y + h + 20 + idx * 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                            # Write the buffered frames to the output video
                            out.write(buffered_frame)

                        # Increment the prediction counter
                        prediction_counter += 1

                        # Clear the list for the next 32-frame block and empty the frame buffer
                        best_face_list = []
                        frame_buffer = []

        # Add the current frame to the buffer
        frame_buffer.append(frame.copy())

        # If the frame buffer size exceeds 96, keep the most recent 96 frames
        if len(frame_buffer) > 96:
            frame_buffer.pop(0)

        # Increment the frame counter
        frame_count += 1

        # Print a message to show progress
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Processing of video {video_path} completed. Output saved to {output_path}")

output_face_dir = r'D:\thesis dataset\split_dataset\test\processed1'
os.makedirs(output_face_dir, exist_ok=True)  # Create directory if it doesn't exist
# Define transformation (same as the training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the model and set it to evaluation mode
num_classes = len(training_dataset.label_map)
model = FineTunedResNet3D(num_classes=num_classes)

# Load the best weights
model.load_state_dict(torch.load('top_model_weights.pth'))

# Set the model to evaluation mode (important for inference)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()



# Example usage
video_path = r'D:\Desktop\thesis\demo\5981354-uhd_4096_2160_25fps.mp4'
output_video_path = r'D:\Desktop\thesis\demo\o_5981354-uhd_4096_2160_25fps.mp4'
process_video_with_bounding_boxes(video_path, 
                                  output_path=output_video_path)
