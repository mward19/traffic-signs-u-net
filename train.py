import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from torchvision import transforms
import albumentations as A
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

from sklearn.neighbors import KernelDensity

from tqdm import tqdm

from model import UNet
from data import SignDataset

from datetime import datetime

import pdb

from collections import Counter

def get_train_val_datasets(seed=42, augment=True):
    # Initialize the train dataset, along with appropriate augmentations
    transform = A.Compose([
        A.RandomCrop(width=512, height=512, p=1.),
        A.Rotate(limit=180, p=0.75),
        A.AdvancedBlur(blur_limit=(9, 17), sigma_x_limit=(0.5, 4), sigma_y_limit=(0.5, 4), p=0.8),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25, p=1.0)
    ], is_check_shapes=False)

    # Collect training data
    train_image_paths = sorted(glob.glob('Traffic Signs/train/images/*.jpg'))
    train_label_paths = sorted(glob.glob('Traffic Signs/train/labels/*.txt'))

    assert len(train_image_paths) == len(train_label_paths)

    # Select data for validation set
    X_train, X_val, y_train, y_val = train_test_split(train_image_paths, train_label_paths, train_size=0.8)

    train_dataset = SignDataset(
        X_train,
        y_train,
        transform
    )

    val_dataset = SignDataset(
        X_val,
        y_val,
        None
    )

    # Initialize the test dataset
    test_dataset = SignDataset(
        glob.glob('Traffic Signs/valid/images/*.jpg'),
        glob.glob('Traffic Signs/valid/labels/*.txt'),
        None
    )

    return train_dataset, val_dataset, test_dataset

def weighted_bce(input, target):
    # Get the weights as a NumPy array and convert it to a tensor on the same device as `input`
    target_cpu = target.cpu().numpy()
    # Weight should be high when prob is near one, low otherwise. Search over params?
    s = 10 # Scale. Really ought to be calculated based on the data
    weights = 1 + s*(target_cpu)**2 
    weights = torch.tensor(weights, dtype=input.dtype, device=input.device)  # Directly create tensor on the correct device
    loss = F.binary_cross_entropy_with_logits(input.squeeze(1), target, weight=weights)
    return loss

def train_model(model, device, num_epochs=300):
    # For now, just train once on the whole train set and validate with the validation set. 
    # Later I'll do better data augmentation, cross-validation, and such
    train_dataset, val_dataset, test_dataset = get_train_val_datasets()
    print(len(train_dataset))

    # Prepare to load data
    train_dataloader = DataLoader(train_dataset, batch_size=4)
    val_dataloader = DataLoader(val_dataset)
    test_dataloader = DataLoader(test_dataset)

    # Prepare loss function weights, weighted by density of values using DenseWeights
    loss_function = weighted_bce
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    patience = 3  # Number of epochs to wait before stopping
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
        total_train_loss = 0
        batches = 0
        for images, masks in pbar:
            # Send to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)

            total_train_loss += loss.item()
            batches += 1
            
            # Backpropagate
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
        
        print(f'Train loss: {total_train_loss / batches:.3f}')

        # Validation phase
        model.eval()
        total_val_loss = 0
        batches = 0  # Count batches to average later
        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                val_loss = loss_function(outputs, masks)
                total_val_loss += val_loss.item()
                batches += 1  # Count batches to average later

        avg_val_loss = total_val_loss / batches
        print(f'Validation loss: {avg_val_loss:.3f}')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset counter if validation loss improves
        else:
            patience_counter += 1  # Increment counter if no improvement
        
        if patience_counter >= patience:
            # Save best model
            torch.save(model.state_dict(), f'model_data/{timestamp}_best_unet_model.pth')
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
        torch.save(model.state_dict(), f'model_data/{timestamp}_unet_model_epoch_{epoch+1}.pth')

    return model

if __name__ == '__main__':
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # I just want to detect signs in general, with probabilities
    n_categories = 1
    model = UNet(3, n_categories).to(device)

    model = train_model(model, device, 300)
