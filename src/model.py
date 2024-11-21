import torch
import torch.nn as nn
from diffusers import UNet2DModel

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class SpriteModel(nn.Module):
    def __init__(self):
        super().__init__()

        num_classes = 5
        class_emb_size = 3

        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=3 + class_emb_size,
            out_channels=3,
            layers_per_block=3,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, noisy_images, timesteps, labels):
        bs, ch, w, h = noisy_images.shape

        class_cond = self.class_emb(labels)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

        total_input = torch.cat((noisy_images, class_cond), 1)

        out = self.model(total_input, timesteps).sample

        return out

    def forward_shapes(self, noisy_images, timesteps, labels):
        bs, ch, w, h = noisy_images.shape
        logger.debug(f"noisy_images.shape : {noisy_images.shape}")
        # torch.Size([4, 3, 16, 16])

        logger.debug(f"labels.shape : {labels.shape}")
        # torch.Size([4])

        # Map to embedding dimension
        class_cond = self.class_emb(labels)
        logger.debug(f"class_cond.shape : {class_cond.shape}")
        # torch.Size([4, 10])  # let class_emb_size = 10

        # class_cond is now (4, 5, 28, 28)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        logger.debug(f"class_cond.shape : {class_cond.shape}")
        # torch.Size([4, 10, 16, 16])

        total_input = torch.cat((noisy_images, class_cond), 1)
        logger.debug(f"total_input.shape : {total_input.shape}")
        # torch.Size([4, 10 + 3, 16, 16])

        out = self.model(total_input, timesteps).sample
        logger.debug(f"out.shape : {out.shape}")
        # torch.Size([4, 3, 16, 16])

        return out

