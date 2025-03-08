import sys
import os
import cv2
import torch
import shutil
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

pytorch_grad_cam_path = os.path.abspath("..")
sys.path.insert(0, pytorch_grad_cam_path)
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image

# Add the cloned kinetics-i3d repo to the path
i3d_path = os.path.abspath("../har_models/kinetics-i3d")
sys.path.insert(0, i3d_path)

from i3d import InceptionI3d

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained I3D model
model = InceptionI3d(400, in_channels=3)  # Assuming Kinetics-400 with 400 classes
model.load_state_dict(torch.load('path_to_pretrained_i3d_model.pth'))  # Load model weights
model = model.to(device)
model.eval()

# Preprocessing function for each frame
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Resize for I3D compatibility
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Adjust for I3D
    ])
    return transform(frame)

# Load video and extract frames into clips (I3D expects temporal sequences)
def load_video(video_path, clip_length=64, frame_sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    rgb_frames = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(frame)  # Save for visualization
            frames.append(preprocess_frame(Image.fromarray(frame)))  # Preprocess frame
        
        frame_id += 1

    cap.release()
    
    # Stack frames into clips for I3D processing
    input_clips = []
    for i in range(0, len(frames) - clip_length + 1, clip_length):
        clip = torch.stack(frames[i:i + clip_length])  # Create a tensor clip
        input_clips.append(clip)

    if not input_clips:
        raise ValueError("Not enough frames to form a single clip!")

    return torch.stack(input_clips), rgb_frames  # Return stacked clips & original frames

# Run I3D model inference on a video clip
def run_inference_on_clip(clip, model):
    clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)  # I3D expects (B, C, T, H, W)
    with torch.no_grad():
        outputs = model(clip)
    return outputs

# Visualize and save Grad-CAM explanations
def visualize_and_save(frames, batch_explanations, output_dir):
    os.makedirs(f"{output_dir}/auxiliary_frames", exist_ok=True)
    visualization_images = []

    for i in range(len(batch_explanations[0])):
        visualization = show_factorization_on_image(
            np.array(frames[i]) / 255.0,  # Normalize frame
            batch_explanations[0][i],  
            image_weight=0.3
        )

        # Convert visualization to image format and save
        visualization_img = Image.fromarray((visualization * 255).astype('uint8'))
        visualization_img.save(os.path.join(f"{output_dir}/auxiliary_frames", f"frame_{i:03d}.jpg"))
        visualization_images.append(visualization_img)

    return visualization_images

# Function to clean up auxiliary frames after processing
def cleanup_auxiliary_frames(output_dir):
    aux_output_dir = os.path.join(output_dir, "auxiliary_frames")
    if os.path.exists(aux_output_dir):
        shutil.rmtree(aux_output_dir)
        print(f"Auxiliary frames cleaned up from '{aux_output_dir}'.")

# Main processing pipeline
dataset_name = "Kinetics-400"
action_name = "throw"
video_path = f"samples/{dataset_name}/{action_name}.mp4"
print(f"Loading video from '{video_path}'...")
input_tensor, rgb_frames = load_video(video_path)

# Use Deep Feature Factorization for interpretability
dff = DeepFeatureFactorization(model=model, target_layer=model.Mixed_5c, computation_on_concepts=model.fc)

# Set number of factorized components
n_components = 10
concepts, batch_explanations, concept_scores = dff(input_tensor, n_components)

# Save visualizations
output_dir = "output_frames"
visualization_images = visualize_and_save(rgb_frames, batch_explanations, output_dir)

# Cleanup auxiliary frames after processing
cleanup_auxiliary_frames(output_dir)
