import torch
import matplotlib.pyplot as plt


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def visualize_X_samples_grid(images, labels, n_samples=12, n_cols=4, filepath=None):
    images_vis = denormalize(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    images_vis = images_vis.reshape(-1, 16, 16, 3)

    if isinstance(images_vis, torch.Tensor):
        images_vis = images_vis.detach().cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    # Populate each subplot
    for i, ax in enumerate(axes.flat):
        ax.axis('off')

        if i >= n_samples:  # Avoid index error if images < n_samples
            break

        image = images_vis[i]
        label = labels[i]

        ax.imshow(image.squeeze())
        ax.set_title(f"Label: {label}")

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close()
