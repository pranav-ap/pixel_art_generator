import torch
import torch.nn as nn
from diffusers import UNet2DModel

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class SpriteModel(nn.Module):
    def __init__(self, num_classes=5, class_emb_size=5):
        super().__init__()

        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=3 + class_emb_size,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, noisy_images, timesteps, labels):
        # x is shape (bs, 3, 28, 28)
        bs, ch, w, h = noisy_images.shape

        # Map to embedding dimension
        class_cond = self.class_emb(labels)
        # class_cond is now (bs, 3, 28, 28)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

        # (bs, 6, 28, 28)
        total_input = torch.cat((noisy_images, class_cond), 1)

        # (bs, 1, 28, 28)
        return self.model(total_input, timesteps).sample
