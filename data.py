"""
Loads in the traffic signs dataset and provides an appropriate dataloader for U-Net training.
"""

import numpy as np
import pandas as pd
import os
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

from tqdm import tqdm

import pdb
#pdb.set_trace = lambda: None

def create_cone_mask(center, image_shape, radius):
    """
    Creates a 2D cone-shaped mask.

    Args:
        center (tuple): (x, y) coordinates of the cone center.
        image_shape (tuple): Shape of the output mask (height, width).
        radius (float): Radius of the cone, beyond which values are zero.

    Returns:
        np.ndarray: A 2D array with the cone-shaped mask.
    """
    x_center, y_center = center

    # Create a grid of coordinates
    x_range = np.arange(image_shape[0])
    y_range = np.arange(image_shape[1])
    x_grid, y_grid = np.meshgrid(x_range, y_range, indexing='ij')

    # Calculate distance from the center
    distance = np.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)

    # Create the cone: Linearly decrease intensity with distance
    cone = np.maximum(0, 1 - (distance / radius))

    return cone

def get_mask(labels, image_shape, cone=True):
    """ Turns bounding boxes into masks for the U-Net. """
    mask = np.zeros(image_shape)
    for row in labels:
        category, pos = row
        # Only one category
        category = 1
        x_pos_norm, y_pos_norm, x_size_norm, y_size_norm = pos
        
        x_pos = int(x_pos_norm * image_shape[0])
        y_pos = int(y_pos_norm * image_shape[1])
    
        x_size = int(x_size_norm * image_shape[0])
        y_size = int(y_size_norm * image_shape[1])
        
        if cone:
            #radius = np.mean([x_size, y_size])
            radius = 50
            cone_kernel = create_cone_mask(center=(x_pos, y_pos), image_shape=image_shape, radius=radius)
            mask = np.maximum(mask, cone_kernel)
        else:
            mask[x_pos - x_size : x_pos + x_size + 1, y_pos - y_size : y_pos + y_size + 1] = category

    return mask.T

if __name__ == "__main__":
    #print(create_cone_mask(np.array([5, 5]), [10, 10], 3))
    test_mask = get_mask([[1, (.4, .6, .1, .1)], [2, (.1, .8, .2, .12)]], (200, 200), cone=True)
    plt.imshow(test_mask)
    plt.savefig('temp/test_mask.png')


class SignDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        """
        Args:
            image_paths (list): List of file paths to the images.
            label_paths (list): List of file paths to the labels.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

        if len(self.image_paths) != len(self.label_paths):
            raise ValueError("The number of image paths and label paths must be the same.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load label
        label_path = self.label_paths[idx]
        with open(label_path, "r") as f:
            labels_raw = [line.split(' ') for line in f.read().strip().split('\n')]

        labels = []
        for label_raw in labels_raw:
            if len(label_raw) != 5:
                continue
            category = int(label_raw[0]) + 1  # Category as integer. Add one so that the lowest category is 1
            bbox = torch.tensor([
                float(label_raw[1]),  # Normalized x position
                float(label_raw[2]),  # Normalized y position
                float(label_raw[3]),  # Normalized width
                float(label_raw[4]),  # Normalized height
            ], dtype=torch.float32)
            labels.append((category, bbox))

        # Load image
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Get label mask
        mask = get_mask(labels, image.shape[:2])

        # Apply transforms to both image and mask
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert image and mask to tensor
        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Return image and label mask (mask as float tensor)
        return image, mask


