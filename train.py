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

from sklearn.neighbors import KernelDensity
from denseweight import DenseWeight

from tqdm import tqdm

from model import UNet
from data import SignDataset

from datetime import datetime

import pdb

from collections import Counter

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
    
def calc_weights(dataset):
    # Sample some training masks
    gen = np.random.default_rng()
    n_sample_masks = int(len(dataset) / 10) # TODO: use bayes to figure out how many samples to take
    sample_items = gen.choice(len(dataset), n_sample_masks, replace=False)
    sample_masks = [dataset[i][1] for i in sample_items] # Extract ground truth masks

    # From the sample masks, extract a few values
    n_sample_vals = int(sample_masks[0].shape[0] * sample_masks[0].shape[1] / 50) # TODO: use bayes to figure out how many samples to take from each mask
    samples = []
    for mask in tqdm(sample_masks, desc='Calculating training weights'):
        flat_mask = mask.flatten()
        sample_indices = gen.choice(len(flat_mask), n_sample_vals, replace=False)
        samples += flat_mask[sample_indices]

    # Calculate weights
    samples = np.array(samples)
    pdb.set_trace()
    dw = DenseWeight(alpha=1.0)
    weights = dw.fit(samples)

    i_range = np.linspace(0, 1)
    plt.plot(i_range, [dw([i]) for i in i_range])
    plt.savefig('temp/weights.png')
    return dw

# Calculate and store mean-squared error loss weights for use in loss function
TRAIN_WEIGHT_FUNC = calc_weights(get_train_val_datasets()[0])

def weighted_mse_loss(input, target):
    # Get the weights as a NumPy array and convert it to a tensor on the same device as `input`
    target_cpu = target.cpu().numpy()
    weights = TRAIN_WEIGHT_FUNC(target_cpu).reshape(target_cpu.shape)  # Convert target to NumPy array on CPU
    weights = torch.tensor(weights, dtype=input.dtype, device=input.device)  # Directly create tensor on the correct device

    # Compute the weighted MSE loss
    #pdb.set_trace()
    loss = F.mse_loss(input.squeeze(1), target, reduction='none')  # Element-wise MSE loss
    return (weights * loss).mean()  # Weighted average of the MSE loss

def train_model(model, device, num_epochs=300):
    # For now, just train once on the whole train set and validate with the validation set. 
    # Later I'll do better data augmentation, cross-validation, and such
    train_dataset, val_dataset, test_dataset = get_train_val_datasets()

    # Prepare to load data
    train_dataloader = DataLoader(train_dataset, batch_size=4)
    val_dataloader = DataLoader(val_dataset)
    test_dataloader = DataLoader(test_dataset)

    # Prepare loss function weights, weighted by density of values using DenseWeights
    loss_function = weighted_mse_loss
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
