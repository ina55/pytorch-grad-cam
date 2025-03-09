import sys 
import os
import cv2
import torch
import shutil
import numpy as np
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F  # Import pentru F.resize
import torch.nn.functional as NF

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
    
def resize_mask_to_original(mask, original_size):
    # If the mask has 3 channels, convert it to a single channel (take the average or the first channel)
    if len(mask.shape) == 3:  # If the mask is 3D (C, H, W)
        mask = mask.mean(axis=0)  # Average over the channels to get a single channel

    if len(mask.shape) == 2:  # Now mask should be 2D
        # Convert the mask to a tensor, add batch and channel dimensions
        mask_resized = NF.interpolate(
            torch.tensor(mask).unsqueeze(0).unsqueeze(0).float(),  # Convert mask to float tensor
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).permute(0, 2, 1).numpy()  # Remove batch and channel dimensions
    else:
        raise ValueError(f"Unexpected shape of the mask: {mask.shape}. Mask should be 2D.")
    
    return mask_resized

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
    
    if min_length > 10:
        min_length = 10
    print(f"Processing {min_length} frames...")
        
    for i in range(min_length):  
        threshold = 0.5
        
        # Use the explanation for the current frame
        batch_explanation_np = np.array(batch_explanations[i])
        
        # Apply the thresholding operation
        batch_explanation_np = np.where(batch_explanation_np > threshold, batch_explanation_np, 0)
        
        # Create visualization
        visualization = show_factorization_on_image(
            np.array(frames[i]) / 255,  # Convert frame to numpy and normalize
            batch_explanation_np[0],  
            image_weight=0.5
        )

        # Convert to PIL Image and save
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
        


def average_frames(rgb_frames, num_frames_target):
    """
    Average the frames into a smaller number of frames.

    :param rgb_frames: List of frames (List of numpy arrays or tensors)
    :param num_frames_target: The number of frames you want to end up with
    :return: A list of averaged frames
    """
    num_frames = len(rgb_frames)
    # Calculate the step size for grouping frames
    step = num_frames // num_frames_target

    # Average groups of frames
    averaged_frames = []
    for i in range(0, num_frames, step):
        # Get the next 'step' number of frames
        group = rgb_frames[i:i + step]
        
        # Ensure group size matches the target step
        if len(group) == step:
            # Average the frames in the group
            averaged_frame = np.mean(group, axis=0)  # Average over the group dimension (axis=0)
            averaged_frames.append(averaged_frame)
    
    return averaged_frames

def average_frames_tensor(input_tensor: torch.Tensor, num_frames_target: int):
    """
    Average the frames in the input tensor to reduce the number of frames to `num_frames_target`.

    :param input_tensor: The input tensor of shape (batch_size, original_num_frames, channels, height, width)
    :param num_frames_target: The desired number of frames after averaging
    :return: The downsampled input tensor with averaged frames
    """
    original_num_frames, channels, height, width = input_tensor.shape

    # Calculate the step size for averaging frames
    step = original_num_frames // num_frames_target
    remaining_frames = original_num_frames % num_frames_target  # Handling remainder frames

    # Create a list to store the averaged frames
    averaged_frames = []

    for i in range(0, original_num_frames, step):
        # Select the group of frames to average
        end = min(i + step, original_num_frames)
        frame_group = input_tensor[i:end]  # Get frames in the current group

        # Average the frames across the selected group
        averaged_frame = frame_group.mean(dim=0)  # Averaging along the frame dimension
        averaged_frames.append(averaged_frame)

    # Handle the remaining frames if they are left out due to rounding
    if remaining_frames > 0:
        # Take the last few frames that were not included
        remaining_group = input_tensor[-remaining_frames:]
        averaged_frame = remaining_group.mean(dim=0)
        averaged_frames.append(averaged_frame)

    # Stack the averaged frames and return the new tensor
    downsampled_tensor = torch.stack(averaged_frames, dim=1)
    
    return downsampled_tensor


# Load video frames
dataset_name = "Kinetics-400"
action_names = sys.argv[1:]

for action_name in action_names:
    video_path = f"samples/{dataset_name}/{action_name}.mp4"
    print(f"Loading video from '{video_path}'...")
    input_tensor, rgb_frames = load_video(video_path)
    
    num_frames_target = 3
    rgb_frames = average_frames(rgb_frames, num_frames_target)
    input_tensor = average_frames_tensor(input_tensor, num_frames_target)    
        
    # Add batch dimension and frames dimension (each frame treated as a sequence)
    input_tensor = input_tensor.unsqueeze(0)  # Adding batch dimension
    
    # Define Deep Feature Factorization
    dff = DeepFeatureFactorization(model=model, target_layer=model.layer3, computation_on_concepts=model.fc)
    
    # Initialize list to store results
    visualization_images = []
    
    n_components = 5
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
