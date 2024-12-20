from config import config
from utils import logger
import torch
import lightning as L
import lightning.pytorch as pl
from src import SpriteLightning, SpriteDataModule
from utils import make_clear_directory


torch.set_float32_matmul_precision('medium')


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    dm = SpriteDataModule()
    light = SpriteLightning()

    trainer = pl.Trainer(
        default_root_dir=config.paths.roots.output,
        logger=L.pytorch.loggers.CSVLogger(save_dir=config.paths.output.logs),
        devices='auto',
        accelerator="auto",
        max_epochs=config.train.max_epochs,
        log_every_n_steps=config.train.log_every_n_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        num_sanity_val_steps=config.train.num_sanity_val_steps,
        enable_model_summary=False,
        fast_dev_run=config.train.fast_dev_run,
        overfit_batches=config.train.overfit_batches,
    )

    trainer.fit(
        light,
        datamodule=dm,
        # ckpt_path='./output/checkpoints/last-v1.ckpt',
    )

    # noinspection PyUnresolvedReferences
    if trainer.checkpoint_callback.best_model_path:
        # noinspection PyUnresolvedReferences
        logger.info(f"Best model path : {trainer.checkpoint_callback.best_model_path}")


def prep_directories():
    logger.info("Clearing Directories")
    make_clear_directory(config.paths.output.logs)
    make_clear_directory(config.paths.output.val_images)
    make_clear_directory(config.paths.output.test_images)


def main():
    torch.cuda.empty_cache()
    prep_directories()

    train()


if __name__ == '__main__':
    main()
