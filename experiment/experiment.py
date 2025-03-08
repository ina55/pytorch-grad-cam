import sys
import os
import cv2
import torch
import shutil
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.video import r3d_18

# Add pytorch_grad_cam path to system path
pytorch_grad_cam_path = os.path.abspath("..")
sys.path.insert(0, pytorch_grad_cam_path)
from pytorch_grad_cam import DeepFeatureFactorization

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained ConvLSTM model, settings may vary
model = r3d_18(pretrained=True)
model = model.to(device)
model.eval()

# Preprocessing function for each frame
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to tensor and automatically handles the channels
        transforms.Resize((224, 224)),  # Resize for I3D compatibility
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Adjust for I3D
    ])
    return transform(frame)

# Load video and extract frames into clips (I3D expects temporal sequences)
def load_video(video_path, clip_length=64, frame_sample_rate=5):
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

    # Stack frames into a sequence:
    input_sequences = torch.stack(frames)

    # Ensure that we have enough frames
    if len(input_sequences) < clip_length:
        # Padding with the last frame if the number of frames is less than required
        padding = clip_length - len(input_sequences)
        input_sequences = torch.cat([input_sequences, input_sequences[-1:].repeat(padding, 1, 1, 1)], dim=0)
    
    return input_sequences

# Main processing pipeline
dataset_name = "Kinetics-400"
action_name = "throw"
video_path = f"samples/{dataset_name}/{action_name}.mp4"
print(f"Loading video from '{video_path}'...")

# Use Deep Feature Factorization for interpretability
dff = DeepFeatureFactorization(model=model, target_layer=model.layer3[0], computation_on_concepts=model.fc)

# Load video and extract frames
input_tensor = load_video(video_path, clip_length=16, frame_sample_rate=1)

input_tensor = input_tensor.permute(1, 0, 2, 3)  # [3, num_frames, 224, 224]
input_tensor = input_tensor.unsqueeze(0)  # [1, 3, num_frames, 224, 224]

# Verify the shape
print(f"Shape of input_tensor after averaging over frames and adding batch dimension: {input_tensor.shape}")
# Expected shape: [1, 3, 224, 224]

# Now, you can pass this tensor to DeepFeatureFactorization
dff = DeepFeatureFactorization(model=model, target_layer=model.layer3[0], computation_on_concepts=model.fc)

n_components = 5
concepts, batch_explanations, concept_scores = dff(input_tensor, n_components)

# Save visualizations
output_dir = "output_frames"
visualization_images = visualize_and_save(input_tensor, batch_explanations, output_dir)

# Cleanup auxiliary frames after processing
cleanup_auxiliary_frames(output_dir)
