from config import config
from utils import logger
import torch
import lightning as L
import lightning.pytorch as pl
from src import SpriteLightning, SpriteDataModule

torch.set_float32_matmul_precision('medium')


def main():
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    checkpoint_path = './output/checkpoints/best-checkpoint-v1.ckpt'
    light = SpriteLightning.load_from_checkpoint(checkpoint_path)
    light.generate()


if __name__ == '__main__':
    main()
