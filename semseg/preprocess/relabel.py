import os
from collections import namedtuple

import cv2
import numpy as np

from ..labels import labels

# Create a mapping from id to trainId
id_to_trainid = {label.id: label.trainId for label in labels if label.id != -1}

# Input and output directories
input_dir = "C:/Users/MED7CLJ/OneDrive - Bosch Group/UBB_thesis/deeplabv3_resnet50/dataset/training/label"
output_dir = "C:/Users/MED7CLJ/OneDrive - Bosch Group/UBB_thesis/deeplabv3_resnet50/dataset/training/label_19"
os.makedirs(output_dir, exist_ok=True)


def process_image(image_path):
    """Map ID to trainId in the given image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    trainid_image = np.full_like(image, 255)  # Default to 255 (ignore label)
    for id_val, train_id in id_to_trainid.items():
        trainid_image[image == id_val] = train_id
    return trainid_image


# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):  # Assuming the files are PNG images
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        trainid_map = process_image(input_path)
        if trainid_map is not None:
            cv2.imwrite(output_path, trainid_map)
            print(f"Processed and saved: {output_path}")
