from config import config
from utils import logger
import torch
from src import SpriteLightning

torch.set_float32_matmul_precision('medium')


def main():
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    checkpoint_path = './output/checkpoints/best_checkpoint-v1.ckpt'
    light = SpriteLightning.load_from_checkpoint(checkpoint_path)
    light.generate(stage='test')


if __name__ == '__main__':
    main()
