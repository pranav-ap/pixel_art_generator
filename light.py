from utils import logger, make_clear_directory
from config import config
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')


class SpriteLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        from model import SpriteModel
        self.model = SpriteModel()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.save_hyperparameters(ignore=['model', 'noise_scheduler'])

        # for dir_key in ['output_val_images', 'output_test_images']:
        #     make_clear_directory(getattr(config.dirs, dir_key))

    def forward(self, noisy_images, timesteps, labels):
        x = self.model(noisy_images, timesteps, labels)
        return x

    def shared_step(self, batch):
        clean_images, labels = batch
        
        noise = torch.randn(clean_images.shape, device=self.device)
        batch_size = clean_images.shape[0]
        timesteps_count = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, timesteps_count, (batch_size,), dtype=torch.int64, device=self.device)

        # noinspection PyTypeChecker
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
    
        noise_pred = self.model(noisy_images, timesteps, labels)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def checkout(self, batch):
        clean_images, labels = batch

        noise = torch.randn(clean_images.shape, device=self.device)
        batch_size = clean_images.shape[0]
        timesteps_count = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, timesteps_count, (batch_size,), dtype=torch.int64, device=self.device)

        # noinspection PyTypeChecker
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict noise

        # logger.debug(f'noisy_images.shape {noisy_images.shape}')
        # logger.debug(f'timesteps {timesteps[:4]}')
        # logger.debug(f'labels {labels[:4]}')

        noise_pred = self.model(noisy_images, timesteps, labels)

        return noise_pred

    def generate(self):
        logger.info('Generating Sample Images')
        samples = torch.randn(config.test.batch_size, 3, config.image_size, config.image_size, device=self.device)
        labels = torch.randint(0, 5, (config.test.batch_size,), dtype=torch.int64, device=self.device)

        timesteps = 500
        self.noise_scheduler.set_timesteps(timesteps, device=self.device)

        for t in tqdm(range(timesteps - 1, -1, -1), desc="Generating images from noise"):
            noise_pred = self.model(samples, timesteps, labels).to(self.device)
            samples = self.noise_scheduler.step(noise_pred, t, samples).prev_sample.to(self.device)

        # Normalize to [0, 1]
        min_val = samples.min()
        max_val = samples.max()
        samples = (samples - min_val) / (max_val - min_val)

        filepath = f"{config.dirs.output_test_images}/samples.npy"
        np.save(filepath, samples.detach().cpu().numpy())
        filepath = f"{config.dirs.output_test_images}/labels.npy"
        np.save(filepath, labels.detach().cpu().numpy())

        return samples, labels

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate
        )

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=2,
                factor=0.5
            ),
            "monitor": "train_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=config.train.patience,
            mode="min",
            verbose=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{config.dirs.output}/checkpoints/',
            filename="best-checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=10)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [checkpoint_callback, early_stop_callback, progress_bar_callback, lr_monitor_callback]
