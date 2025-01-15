import numpy as np
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt
from train import get_train_val_datasets
import torch
from torch.utils.data import Subset
from model import UNet
from tqdm import tqdm

import pdb

def viz_eval(model, dataset):
    """ Find and visualize the model's predictions for some dataset """
    # Select random data
    for i, (image, mask) in tqdm(enumerate(dataset)):
        image = image.reshape(1, *image.shape)
        
        # Make prediction
        pred = model(image)

        # Convert to numpy ndarrays for plotting
        image = image.numpy()
        pred = pred.detach().numpy()[0, 0] # There's only one sample
        # Extract maximum points from prediction image. TODO: min_distance and more especially threshold_abs are set arbitrarily but could benefit from a more intelligent selection
        pred_points = peak_local_max(pred, min_distance=30, threshold_abs=1, p_norm=2)

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot real image
        ax[0].imshow(np.transpose(image.squeeze(0), (1, 2, 0)))

        # Plot predictions over it
        ax[0].scatter(
            pred_points[:, 1], 
            pred_points[:, 0], 
            facecolors='yellow', 
            edgecolors='white', 
            linewidths=2, 
            s=100, 
            marker='*'
        )
        
        mask_image = ax[1].imshow(pred, cmap='viridis')
        fig.colorbar(mask_image, ax=ax[1], orientation='vertical', fraction=0.05, pad=0.04)

        fig.savefig(f'model_eval/model_eval_{i}.png')

if __name__ == '__main__':
    print("Beginning evaluation...")
    model = UNet(3, 1)
    model.load_state_dict(torch.load('/home/mward19/Documents/kaggle_traffic_signs/model_data/20250114_201149_unet_model_epoch_15.pth', weights_only=True))
    model.eval()

    train_dataset, val_dataset, test_dataset = get_train_val_datasets(augment=False)
    viz_eval(model, Subset(test_dataset, range(25, 35)))