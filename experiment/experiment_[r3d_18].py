import sys 
import os
import cv2
import torch
import shutil
import numpy as np
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Add pytorch_grad_cam path to system path
pytorch_grad_cam_path = os.path.abspath("..")
sys.path.insert(0, pytorch_grad_cam_path)
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the r3d_18 model (ResNet-3D with 18 layers)
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
    rgb_frames = []  # Store original frames for visualization

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            rgb_frames.append(frame)  # Save for visualization
            frames.append(preprocess_frame(Image.fromarray(frame)))

        frame_id += 1

    cap.release()
    return torch.stack(frames), rgb_frames

# Define visualization function
def visualize_and_save(frames, batch_explanations, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    visualization_images = []

    min_length = min(len(frames), len(batch_explanations[0]))
    if (min_length > 10):
        min_length = 10
    print(f"Processing {min_length} frames...")
    
    threshold = 0.5
    # Convert batch_explanations[0] to a numpy array
    batch_explanations_np = np.array(batch_explanations[0])
    
    # Now apply the thresholding operation
    batch_explanations_np = np.where(batch_explanations_np > threshold, batch_explanations_np, 0)
    batch_explanations[0] = batch_explanations_np
    
    for i in range(min_length):  
        mask = np.zeros_like(frames[1])  # Create a blank mask with the same size as the frame
        visualization = show_factorization_on_image(
            np.array(frames[i]) / 255.0,  # Convert frame to numpy
            batch_explanations[0][i],  
            image_weight=0.7
        )

        # Convert visualization to image format and save
        visualization_img = Image.fromarray((visualization * 255).astype('uint8'))
        if not os.path.exists(f"{output_dir}/auxiliary_frames"):
            os.makedirs(f"{output_dir}/auxiliary_frames", exist_ok=True)
        visualization_img.save(os.path.join(f"{output_dir}/auxiliary_frames", f"frame_{i:03d}.jpg"))
        visualization_images.append(visualization_img)

    return visualization_images

# Define function to concatenate images horizontally
def concatenate_images(image_list, output_path):
    if not image_list:
        print("No images to concatenate.")
        return

    widths, heights = zip(*(img.size for img in image_list))
    total_width = sum(widths)
    max_height = max(heights)

    concatenated_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in image_list:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    concatenated_image.save(output_path)
    print(f"Final concatenated image saved as '{output_path}'")
    
def cleanup_auxiliary_frames(output_dir):
    # Remove all files in the auxiliary frames directory
    aux_output_dir = f"{output_dir}/auxiliary_frames"
    if os.path.exists(aux_output_dir):
        shutil.rmtree(aux_output_dir)
        print(f"Auxiliary frames cleaned up from '{aux_output_dir}'.")
    else:
        print(f"No auxiliary frames directory found at '{aux_output_dir}'.")

# Load video frames
dataset_name = "Kinetics-400"
action_names = sys.argv[1:]

for action_name in action_names:
    video_path = f"samples/{dataset_name}/{action_name}.mp4"
    print(f"Loading video from '{video_path}'...")
    input_tensor, rgb_frames = load_video(video_path)
    print(f"Number of frames: {len(rgb_frames)}")
    
    # Add batch dimension and frames dimension (each frame treated as a sequence)
    input_tensor = input_tensor.unsqueeze(0)  # Adding batch dimension
    input_tensor = input_tensor.permute(0, 2, 1, 3, 4)  # Rearranging dimensions
    # print(f"Video loaded with shape: {input_tensor.shape}")
    
    # Define Deep Feature Factorization
    dff = DeepFeatureFactorization(model=model, target_layer=model.layer3, computation_on_concepts=model.fc)
    
    # Initialize list to store results
    visualization_images = []
    
    n_components = 3
    concepts, batch_explanations, concept_scores = dff(input_tensor, n_components)
    
    # Save visualizations
    output_dir = "output_frames"
    visualization_images = visualize_and_save(rgb_frames, batch_explanations, output_dir)
    
    # Concatenate all result photos horizontally
    final_output_filename = f"final_output_{dataset_name}_{action_name}.jpg"
    if not os.path.exists("output_frames/r3d_18"):
        os.makedirs("output_frames/r3d_18")
    final_output_path = os.path.join("output_frames/r3d_18", final_output_filename)
    concatenate_images(visualization_images, final_output_path)
    
    # Cleanup auxiliary frames after processing
    cleanup_auxiliary_frames("output_frames")
