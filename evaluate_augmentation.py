from train import get_train_val_datasets
from matplotlib import pyplot as plt
import numpy as np
import pdb

def see_image_and_mask(dataset, index):
    # Load an image and its mask
    image, mask = dataset[index]
    # Get them in the right form for pyplot
    image, mask = image.numpy(), mask.numpy()
    image = np.permute_dims(image, (1, 2, 0))

    # Plot
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(mask)
    fig.savefig(f'temp/image_and_mask_{index}.png')


if __name__ == '__main__':
    train_set, val_set, _ = get_train_val_datasets()
    for index in range(25, 45):
        see_image_and_mask(train_set, index)