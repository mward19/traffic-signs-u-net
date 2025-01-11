import numpy as np
from matplotlib import pyplot as plt
from train import get_train_val_datasets
import torch
from torch.utils.data import Subset
from model import UNet

def viz_eval(model, dataset):
    """ Find and visualize the model's predictions for some dataset """
    # Select random data
    for i, (image, mask) in enumerate(dataset):
        image = image.reshape(1, *image.shape)
        
        # Make prediction
        pred = model(image)

        # Convert to numpy ndarrays for plotting
        image = image.numpy()
        pred = pred.detach().numpy()[0, 0] # There's only one sample

        fig, ax = plt.subplots(1, 2)
        
        # Plot real image
        ax[0].imshow(image.squeeze(0).transpose(1, 2, 0))

        # Plot prediction
        im = ax[1].imshow(pred, cmap='viridis')
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)

        fig.savefig(f'temp/model_eval_{i}.png')

if __name__ == '__main__':
    print("Beginning evaluation...")
    model = UNet(3, 1)
    model.load_state_dict(torch.load('/home/mward19/Documents/kaggle_traffic_signs/model_data/20250110_173300_unet_model_epoch_10.pth', weights_only=True))
    model.eval()

    train_dataset, val_dataset, test_dataset = get_train_val_datasets()
    viz_eval(model, Subset(test_dataset, range(25, 30)))