import torch

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')


def visualize_X_samples_grid(images, labels, n_samples=12, n_cols=4):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    images_vis = images * std + mean    
    images_vis = images_vis.reshape(-1, 16, 16, 3)
    
    n_rows = n_samples // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for i, ax in enumerate(axes.flat):
        label = labels[i].argmax().item()
        img = images_vis[i]

        if isinstance(img, torch.Tensor):  
            img = img.detach().numpy()  

        ax.imshow(img.squeeze())
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

