
### Steps
1. Load a pre-trained model

    We’re using r3d_18, a 3D ResNet model trained for video action recognition.
    The model takes a video as input and extracts features that help recognize actions.

2. Load a video and extract frames

   The video is read, and frames are extracted every 5 frames (to reduce processing time).
   The frames are normalized and converted to a tensor for the model.

3. Apply Deep Feature Factorization (DFF)

   DFF is applied to the last convolutional layer (model.layer4) to decompose model features into components.
   This means it tries to find meaningful patterns in how the model sees the video.

    > **DFF helps break down model activations into smaller, more interpretable components.**
    >
    > 1️⃣ The model processes the video and extracts feature maps at the last layer.  
    > 2️⃣ These feature maps contain a lot of mixed information about objects, movement, and background.  
    > 3️⃣ DFF factorizes (decomposes) these feature maps into independent parts using Non-Negative Matrix Factorization (NMF).  
    > 4️⃣ This gives separate feature components, each highlighting different aspects of the video:
    >    - One component might focus on motion
    >    - Another on objects
    >    - Another on edges or background  
    > 5️⃣ The extracted components are visualized by overlaying them on the frames.
             
4. Visualize the results

   The extracted features are overlaid on the original frames using colors.
   The images are saved, concatenated, and cleaned up.

### Understanding r3d_18

#### What Does `layer3` Do in `r3d_18`?

`r3d_18` (ResNet-3D with 18 layers) follows a **hierarchical feature extraction process**:

| Layer   | Function |
|---------|----------|
| `layer1` | Detects simple edges, corners, and textures. |
| `layer2` | Captures local object parts and small movements. |
| `layer3` | **Focuses on mid-level motion patterns** and object structures. |
| `layer4` | Detects high-level motion semantics and full action patterns. |


### Python Setup & Required packages
`py -3.9 -m venv myenv`
`myenv\Scripts\activate`
`pip install tensorflow==2.6.0 tensorflow-probability==0.14.1 dm-sonnet==2.0.0`
`pip install opencv-python`
`pip install torch torchvision`
`pip install ttach`
`pip install matplotlib`
`pip install scipy`
`pip install scikit-learn`
`pip install tqdm`

### Running the code
`python experiment.py`

### TODO
- Evaluate for each video
- https://github.com/pytorch/vision/blob/main/references/video_classification/README.md
- Preprocess videos before dff
