from config import config
from utils import logger
import torch
import torch.nn as nn
from diffusers import UNet2DModel

torch.set_float32_matmul_precision('medium')


class SpriteModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=4,
            out_channels=3,
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        label_dim = 5
        self.label_embedding = nn.Linear(label_dim, config.image_size * config.image_size)

    def forward(self, noisy_images, labels, timesteps):
        labels_embedded = self.label_embedding(labels).view(-1, 1, config.image_size, config.image_size)
        noisy_images_l = torch.cat([noisy_images, labels_embedded], dim=1)
        noise_pred = self.model(noisy_images_l, timesteps, return_dict=False)[0]
        
        return noise_pred

    def forward_clean(self, noisy_images, labels, timesteps):
        labels_embedded = self.label_embedding(labels).view(-1, 1, config.image_size, config.image_size)
        noisy_images_l = torch.cat([noisy_images, labels_embedded], dim=1)
        images = self.model(noisy_images_l, timesteps).sample
        
        return images
        
