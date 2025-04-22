from typing import List, Dict
import os
import numpy as np
import json


def preprocess_data(dataset: NuScenes, output_dir: str) -> List[Dict]:
    """Preprocess the NuScenes data for training"""
    os.makedirs(output_dir, exist_ok=True)
    samples = load_data(dataset, output_dir)
    processed_data = []

    for sample in samples:
        # Preprocess each sample and save it
        processed_sample = preprocess_sample(sample)
        processed_data.append(processed_sample)

    return processed_data


def preprocess_sample(sample: Dict) -> Dict:
    """Preprocess a single sample"""
    # Example of how we could preprocess an image for VectorBC
    image = sample["data"]["CAM_FRONT"]  # Example: Use front camera image from NuScenes
    image_data = np.array(image)  # Converting image data to NumPy array

    # Preprocessing for VectorBC (reshape, normalize, etc.)
    image_data = preprocess_image(image_data)

    return {"image": image_data}


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess an image for behavior cloning (e.g., resize, normalize)"""
    # Example preprocessing (resize, normalize, etc.)
    return image  # Modify this to suit your needs (e.g., resizing, normalization)

