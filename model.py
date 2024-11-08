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
            sample_size=config.image_size, 
            in_channels=3,
            out_channels=3,
            num_class_embeds=5,
            layers_per_block=2,  
            block_out_channels=(128, 128, 256, 256, 512),
            down_block_types=(
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
            ),            
        )

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, noisy_images, timesteps, labels):
        noise_residual_pred = self.model(noisy_images, timesteps, labels).sample
        # noise_residual_pred = self.model(noisy_images, timesteps).sample
        return noise_residual_pred
