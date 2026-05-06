# Emotion Recognition from Facial Image Sequences using 3D Convolutional Neural Networks

This repository contains my diploma thesis developed at the Department of Electrical and Computer Engineering of the Aristotle University of Thessaloniki. The project focuses on emotion recognition from facial video sequences using deep learning and 3D Convolutional Neural Networks (3D-CNNs).

## Project Overview

The goal of this thesis is to investigate whether human emotional states can be accurately recognized from sequences of facial images extracted from videos.

Unlike traditional approaches that analyze single static images, this project focuses on temporal facial information by processing sequences of frames using 3D convolutional neural networks. The model is inspired by the ResNet3D-18 architecture commonly used in action recognition tasks.

The implemented pipeline includes:
- Video frame extraction
- Face detection and face extraction
- Dataset preprocessing and augmentation
- Emotion classification using deep learning
- Fine-tuning of pretrained 3D CNN architectures
- Model evaluation using multiple metrics

The final model achieved accuracy scores above 83% on the validation dataset.

---

## Motivation

Emotion recognition has applications in:
- Human-computer interaction
- Healthcare and psychology
- Security systems
- Smart environments
- Robotics and AI assistants

The project focuses on recognizing emotions in realistic scenarios where facial expressions evolve dynamically over time rather than remaining static.

---

## Datasets Used

### AFEW-VA
A video-based dataset containing facial expressions annotated in the valence-arousal emotional space.

### K-EMOCON
A multimodal emotional conversation dataset used for additional evaluation and experimentation.

The datasets were processed by:
- Sampling video frames
- Extracting facial regions
- Normalizing and augmenting image sequences

---

## Preprocessing Pipeline

The preprocessing stage includes:
- Face detection and extraction
- Frame sampling at 10 FPS
- Label transformation (discretization)
- Image enhancement
- Brightness and contrast augmentation
- Random rotations
- Horizontal and vertical flipping

These augmentations were applied consistently across video frame sequences to preserve temporal information.

---

## Model Architecture

The implemented model is based on:
- 3D Convolutional Neural Networks (3D-CNNs)
- Residual Networks (ResNet3D-18)
- Fine-tuning of pretrained weights

The architecture captures:
- Spatial facial features
- Temporal changes between consecutive frames
- Dynamic emotional patterns

Unlike 2D CNNs, the 3D model processes full frame sequences and learns temporal dependencies between expressions.

---

## Training and Evaluation

The project includes:
- Supervised training
- Validation experiments
- Hyperparameter tuning
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

The best-performing model achieved validation accuracy above 83%.

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- Deep Learning
- 3D Convolutional Neural Networks
- Computer Vision

---

## Topics Covered

- Emotion Recognition
- Deep Learning
- Computer Vision
- Facial Expression Analysis
- Video Understanding
- 3D CNNs (ResNet3D)
- Face Detection
- Sequence Classification

---

## Thesis Contribution

The thesis investigates how temporal facial information can improve emotion recognition compared to static image approaches.

The implemented methodology:
- Extracts facial sequences from videos
- Processes temporal facial changes
- Fine-tunes a 3D CNN architecture
- Evaluates emotion recognition performance on realistic datasets

The results demonstrate that temporal modeling significantly improves emotion recognition performance in video-based scenarios.

---

## Notes

This project was developed as part of my diploma thesis and focuses on deep learning approaches for emotion recognition from facial video sequences in realistic environments.
