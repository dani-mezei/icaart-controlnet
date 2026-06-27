import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from semseg.labels import labels
from semseg.model import DeeplabV3Resnet101

# Load the fine-tuned model checkpoint
checkpoint_path = "C:/Users/MED7CLJ/OneDrive - Bosch Group/UBB_thesis/thesis/test_weights/step-checkpoint-epoch=03-step=003000.ckpt"
model = DeeplabV3Resnet101.load_from_checkpoint(checkpoint_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Put the model in evaluation mode
model.eval()
model.freeze()  # Ensures no gradients are tracked

# Define preprocessing pipeline
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize
    ]
)

# Load an image
image_path = "/fs/scratch/Clj2_BEG_ens_gpu_cluster/med7clj/KITTI-360/dataset/validation/exp_output/2013_05_28_drive_0000_sync_0000000386.png"
image = Image.open(image_path).convert("RGB")

# Preprocess the image
input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Extract the predicted mask
output_predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# Create a colored mask using the provided label colors
colored_mask = np.zeros(
    (output_predictions.shape[0], output_predictions.shape[1], 3), dtype=np.uint8
)

id_to_label = {label.trainId: label for label in labels}

# Map the predicted class indices to the corresponding color
for class_id in np.unique(output_predictions):
    if class_id in id_to_label:
        color = id_to_label[class_id].color
        colored_mask[output_predictions == class_id] = color

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask")
plt.imshow(colored_mask)
plt.show()
