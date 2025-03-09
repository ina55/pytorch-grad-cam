import sys 
import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model (assuming it is a state_dict model)
model = models.video.r3d_18(pretrained=True).to(device)
model.eval()

# Define preprocessing function
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(frame)

# Load video and extract frames
def load_video(video_path, frame_sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(preprocess_frame(Image.fromarray(frame)))

        frame_id += 1

    cap.release()
    return torch.stack(frames)

# Define function to process video and evaluate model
def process_video(model, video_path):
    # Load video frames
    print(f"Loading video from '{video_path}'...")
    input_tensor = load_video(video_path)
    
    # Add batch dimension and frames dimension (each frame to be treated as a sequence)
    input_tensor = input_tensor.unsqueeze(0)  # Adding batch dimension
    input_tensor = input_tensor.permute(0, 2, 1, 3, 4)  # Rearranging dimensions
    
    # Move input to correct device
    input_tensor = input_tensor.to(device)
    
    # Forward pass through the model 
    with torch.no_grad():
        output = model(input_tensor)
    
    # If it's multiclass classification problem, get the class with the highest prediction
    _, predicted_class = torch.max(output, 1)

    return predicted_class

dataset_name = "Kinetics-400"
action_name = "football"
video_path = f"samples/{dataset_name}/{action_name}.mp4"
predicted_class = process_video(model, video_path)

# load class mappings
class_mappings = pd.read_csv('samples/Kinetics-400/mappings/action_classes.csv')

# Convert DataFrame to dictionary
class_dict = class_mappings.set_index('id')['name'].to_dict()

# Get the human-readable name of the predicted class
predicted_class_name = class_dict[predicted_class.item()]

print(f"Predicted class for video: {predicted_class_name}")
