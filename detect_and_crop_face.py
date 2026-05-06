import cv2
import os
from pathlib import Path
from deepface import DeepFace
import numpy as np

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

# Function to process video and extract frames at 10 FPS
def process_video(video_path, dest_dir, target_fps=10):
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original Frame rate: {fps} FPS")

    # Calculate the interval between frames to achieve the target FPS
    frame_interval = int(round(fps / target_fps))

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image_rgb = preprocess_image(frame)
            if image_rgb is None:
                continue

            # Extract faces from the image with enforce_detection set to False
            detected_faces = DeepFace.extract_faces(
                img_path=image_rgb,
                enforce_detection=True,
                detector_backend='yolov8'
            )

            if detected_faces is not None and len(detected_faces) > 0:
                # Initialize variables to store the highest confidence face and its score
                max_confidence = -1
                best_face = None

                for face_data in detected_faces:
                    face_image = face_data['face']
                    confidence = face_data['confidence']

                    # Check if this face has higher confidence than the current maximum
                    if confidence >= max_confidence:
                        max_confidence = confidence
                        best_face = face_image

                # Check if a valid face with the highest confidence was found
                if best_face is not None:
                    # Convert best_face to uint8 and RGB format if necessary
                    if best_face.dtype != np.uint8:
                        best_face = (255 * (best_face - best_face.min()) / (best_face.max() - best_face.min())).astype(
                            np.uint8)

                    # Save the best face image from the video
                    dest_file_path = os.path.join(dest_dir, f"000{saved_frame_count}.png")

                    # Save the image in RGB format
                    if best_face.shape[-1] == 3:  # Ensure it's RGB format
                        # Save the image in RGB format
                        cv2.imwrite(dest_file_path,best_face)

                        print(f"Saved frame {saved_frame_count} with highest confidence face")
                    else:
                        print(f"Invalid image format for saving: {best_face.shape}")

                    saved_frame_count += 1
                else:
                    print(f"No valid face found in frame {frame_count}")
            else:
                print(f"No face detected in frame {frame_count}")

        frame_count += 1

    cap.release()
    print(f"Processing of video {video_path} completed.")

# Example usage
video_path = r'D:\thesis dataset\split_dataset\test\6189268-hd_1638_1080_25fps.mp4'
dest_dir = r'D:\thesis dataset\split_dataset\test\processed1'
Path(dest_dir).mkdir(parents=True, exist_ok=True)

process_video(video_path, dest_dir, target_fps=10)
