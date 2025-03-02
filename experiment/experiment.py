import torchvision.models as models
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

# Define the model
model = models.resnet50(pretrained=True)
model.eval()

# Load your image
img_path = '../examples/both.png'
img_pil = Image.open(img_path)

# Convert image to a numpy array and normalize it into 0-1 range
rgb_img_float = np.array(img_pil) / 255.0

# Preprocess your input for your model
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(img_pil)
input_tensor = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Move the input tensor to the GPU if one is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

# Define DeepFeatureFactorization
dff = DeepFeatureFactorization(model=model, target_layer=model.layer4, computation_on_concepts=model.fc)

# Set the number of components you want to extract
n_components = 5
concepts, batch_explanations, concept_scores = dff(input_tensor, n_components)

# Visualize the factorization on the image
visualization = show_factorization_on_image(rgb_img_float, 
                                            batch_explanations[0],
                                            image_weight=0.3)
# Convert to uint8 (required by PIL Image) and create an image
visualization_img = Image.fromarray((visualization * 255).astype('uint8'))

# Save the image
visualization_img.save("output.jpg")