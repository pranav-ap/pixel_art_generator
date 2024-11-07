import torch

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')


def visualize_X_samples_grid(images, labels, n_samples=12, n_cols=4):
    images_vis = images.reshape(-1, 16, 16, 3)
    
    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for i, ax in enumerate(axes.flat):
        label = labels[i].item()
        image = images_vis[i].detach().numpy()  

        ax.imshow(image.squeeze())
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

