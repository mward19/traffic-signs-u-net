import numpy as np
import pandas as pd
import os
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

from torchvision import transforms

from tqdm import tqdm

from model import UNet
from data import SignDataset

from datetime import datetime

import pdb

from collections import Counter

def calculate_class_imbalance(dataset, device):
    class_counts = Counter()
    
    for _, mask in tqdm(dataset, desc="Calculating class imbalance"):
        unique, counts = torch.unique(mask, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            class_counts[u] += c
    
    total_pixels = sum(class_counts.values())
    class_weights = {cls: total_pixels / count for cls, count in class_counts.items()}
    
    # Convert class_weights to 1D tensor
    class_weights = torch.tensor([class_weights[key] for key in sorted(class_weights)])
    return class_counts, class_weights.to(device)

def get_train_val_datasets(seed=42):
    # Initialize the dataset
    train_dataset_all = SignDataset(
        'Traffic Signs/train/images',
        'Traffic Signs/train/labels',
        None
    )
    test_dataset = SignDataset(
        'Traffic Signs/valid/images',
        'Traffic Signs/valid/labels',
        None
    )
    
    # Random seed for reproducibility
    torch.manual_seed(seed)

    val_size = int(0.2 * len(train_dataset_all))
    train_size = len(train_dataset_all) - val_size

    train_dataset, val_dataset = random_split(train_dataset_all, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

def train_model(model, device, num_epochs=300):
    # For now, just train once on the whole train set and validate with the validation set. 
    # Later I'll do better data augmentation, cross-validation, and such
    train_dataset, val_dataset, test_dataset = get_train_val_datasets()

    # Prepare to load data
    train_dataloader = DataLoader(train_dataset, batch_size=4)
    val_dataloader = DataLoader(val_dataset)
    test_dataloader = DataLoader(test_dataset)

    # Calculate class imbalance on training dataset
    train_class_counts, train_class_weights = calculate_class_imbalance(train_dataset, device)
    train_class_weights = train_class_weights.to(device)  # Ensure weights are on the same device
    loss_function = nn.CrossEntropyLoss(weight=train_class_weights)
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

    # I just want to detect signs in general. Background / sign
    n_categories = 2
    model = UNet(3, n_categories).to(device)

    model = train_model(model, device, 300)