import numpy as np
from PIL import Image
import torch
from typing import Callable, List, Tuple, Optional
from sklearn.decomposition import NMF
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, create_labels_legend, show_factorization_on_image


def dff(activations: np.ndarray, n_components: int = 5):
    """ Compute Deep Feature Factorization on a 2d Activations tensor.

    :param activations: A numpy array of shape batch x channels x height x width
    :param n_components: The number of components for the non negative matrix factorization
    :returns: A tuple of the concepts (a numpy array with shape channels x components),
              and the explanation heatmaps (a numpy arary with shape batch x height x width)
    """

    batch_size, channels, h, w = activations.shape
    reshaped_activations = activations.transpose((1, 0, 2, 3))
    reshaped_activations[np.isnan(reshaped_activations)] = 0
    reshaped_activations = reshaped_activations.reshape(
        reshaped_activations.shape[0], -1)
    offset = reshaped_activations.min(axis=-1)
    reshaped_activations = reshaped_activations - offset[:, None]

    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(reshaped_activations)
    H = model.components_
    concepts = W + offset[:, None]
    explanations = H.reshape(n_components, batch_size, h, w)
    explanations = explanations.transpose((1, 0, 2, 3))
    return concepts, explanations


class DeepFeatureFactorization:
    """ Deep Feature Factorization: https://arxiv.org/abs/1806.10206
        This gets a model andcomputes the 2D activations for a target layer,
        and computes Non Negative Matrix Factorization on the activations.

        Optionally it runs a computation on the concept embeddings,
        like running a classifier on them.

        The explanation heatmaps are scalled to the range [0, 1]
        and to the input tensor width and height.
     """

    def __init__(self,
                 model: torch.nn.Module,
                 target_layer: torch.nn.Module,
                 reshape_transform: Callable = None,
                 computation_on_concepts=None
                 ):
        self.model = model
        self.computation_on_concepts = computation_on_concepts
        self.activations_and_grads = ActivationsAndGradients(
            self.model, [target_layer], reshape_transform)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 n_components: int = 16):
        # Get the shape of the input tensor: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, h, w = input_tensor.size()
    
        # Apply activations and gradients to the entire input tensor (batch_size x num_frames)
        _ = self.activations_and_grads(input_tensor)
    
        # Initialize a list to hold the concepts and explanations for each frame
        all_concepts = []
        all_processed_explanations = []
    
        with torch.no_grad():
            # Loop through the frames
            # Get activations for the specific frame (indexing the activations per frame)
            activations = self.activations_and_grads.activations[0][0].cpu().numpy()

            # Perform Deep Feature Factorization on the activations of this frame
            concepts, explanations = dff(activations, n_components=n_components)

            # Process the explanation heatmaps for this frame
            processed_explanations = []
            for batch in explanations:
                processed_explanations.append(scale_cam_image(batch, (w, h)))

            all_concepts.append(concepts)
            all_processed_explanations.append(processed_explanations)
    
            # If there is a computation to run on the concepts, apply it here
            if self.computation_on_concepts:
                with torch.no_grad():
                    # Ensure the correct shape for the computation on concepts
                    all_concept_tensors = [torch.from_numpy(np.float32(concept).transpose((1, 0))) for concept in all_concepts]
            
                    # Flatten the concepts to match the expected input shape of the Linear layer (if needed)
                    flattened_concept_tensors = [concept_tensor.reshape(-1) for concept_tensor in all_concept_tensors]
            
                    # Iterate over each concept tensor and apply padding if needed
                    concept_outputs = []
                    for concept_tensor in flattened_concept_tensors:
                       # Check if the tensor is 1D, in that case reshape it to 2D (1, N)
                       if concept_tensor.dim() == 1:
                           concept_tensor = concept_tensor.unsqueeze(0)  # Convert from (N,) to (1, N)
                   
                       # Now apply padding if the tensor size is less than 512
                       if concept_tensor.size(1) < 512:
                           padding = 512 - concept_tensor.size(1)
                           # Pad the tensor with zeros to match the required size
                           padding_tensor = torch.zeros(1, padding, device=concept_tensor.device)  # Ensure it's on the same device
                           padded_concept_tensor = torch.cat([concept_tensor, padding_tensor], dim=1)  # Concatenate along dim=1 (columns)
                       else:
                           padded_concept_tensor = concept_tensor
                   
                       # Now pass the padded tensor through the computation_on_concepts
                       concept_output = self.computation_on_concepts(padded_concept_tensor.reshape(1, 512))
                       concept_outputs.append(concept_output.cpu().numpy())

            
                return all_concepts, all_processed_explanations, concept_outputs

    
            else:
                return all_concepts, all_processed_explanations

    def __del__(self):
        self.activations_and_grads.release()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in ActivationSummary with block: {exc_type}. Message: {exc_value}")
            return True


def run_dff_on_image(model: torch.nn.Module,
                     target_layer: torch.nn.Module,
                     classifier: torch.nn.Module,
                     img_pil: Image,
                     img_tensor: torch.Tensor,
                     reshape_transform=Optional[Callable],
                     n_components: int = 5,
                     top_k: int = 2) -> np.ndarray:
    """ Helper function to create a Deep Feature Factorization visualization for a single image.
        TBD: Run this on a batch with several images.
    """
    rgb_img_float = np.array(img_pil) / 255
    dff = DeepFeatureFactorization(model=model,
                                   reshape_transform=reshape_transform,
                                   target_layer=target_layer,
                                   computation_on_concepts=classifier)

    concepts, batch_explanations, concept_outputs = dff(
        img_tensor[None, :], n_components)

    concept_outputs = torch.softmax(
        torch.from_numpy(concept_outputs),
        axis=-1).numpy()
    concept_label_strings = create_labels_legend(concept_outputs,
                                                 labels=model.config.id2label,
                                                 top_k=top_k)
    visualization = show_factorization_on_image(
        rgb_img_float,
        batch_explanations[0],
        image_weight=0.3,
        concept_labels=concept_label_strings)

    result = np.hstack((np.array(img_pil), visualization))
    return result

