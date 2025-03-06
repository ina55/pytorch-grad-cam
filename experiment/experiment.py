import torchvision.models as models
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image
from torchvision import transforms
import torch
import cv2
import numpy as np
from PIL import Image
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained model (ResNet-50 for now, can replace with I3D, SlowFast, etc.)
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Define video preprocessing function
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
    rgb_frames = []  # Store original frames for visualization

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every 'frame_sample_rate' frames
        if frame_id % frame_sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            rgb_frames.append(frame)  # Save for visualization
            frames.append(preprocess_frame(Image.fromarray(frame)))

        frame_id += 1

    cap.release()
    return torch.stack(frames), rgb_frames

# Load video frames
video_path = "samples/Kinetics-400/throw.mp4"
input_tensor, rgb_frames = load_video(video_path)

# Add batch dimension
input_tensor = input_tensor.unsqueeze(0).to(device)  
input_tensor = input_tensor.mean(dim=1)

# Define Deep Feature Factorization
dff = DeepFeatureFactorization(model=model, target_layer=model.layer4, computation_on_concepts=model.fc)

# Set number of components for factorization
n_components = 10
concepts, batch_explanations, concept_scores = dff(input_tensor, n_components)

# Save visualizations for each frame
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Number of explanations available
num_explanations = len(batch_explanations[0])

# Save visualizations for available explanations only
for i in range(num_explanations):  # Loop over available explanations
    visualization = show_factorization_on_image(
        np.array(rgb_frames[i]) / 255.0,  # Normalize image to [0,1]
        batch_explanations[0][i],  # Get explanation only for valid indices
        image_weight=0.3
    )

    # Convert visualization to image format
    visualization_img = Image.fromarray((visualization * 255).astype('uint8'))
    visualization_img.save(os.path.join(output_dir, f"frame_{i:03d}.jpg"))

print(f"Saved visualized frames in '{output_dir}'")

