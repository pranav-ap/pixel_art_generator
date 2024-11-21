import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from tqdm import tqdm

from config import config
from utils import visualize_X_samples_grid, logger
from .model import SpriteModel

torch.set_float32_matmul_precision('medium')


class SpriteLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SpriteModel()
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=500)

        self.save_hyperparameters(ignore=['model', 'noise_scheduler'])

    def forward(self, noisy_images, timesteps, labels):
        x = self.model(noisy_images, timesteps, labels)
        return x

    def _prepare_inputs(self, batch):
        clean_images, labels = batch

        noise = torch.randn_like(clean_images, device=self.device)
        batch_size = clean_images.shape[0]
        # noinspection PyUnresolvedReferences
        timesteps_count = self.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, timesteps_count, (batch_size,), dtype=torch.int64, device=self.device)

        # noinspection PyTypeChecker
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        return noisy_images, timesteps, labels, noise

    @torch.no_grad()
    def forward_shapes(self, batch):
        noisy_images, timesteps, labels, noise = self._prepare_inputs(batch)
        noise_pred = self.model.forward_shapes(noisy_images, timesteps, labels)
        return noise_pred

    def shared_step(self, batch):
        noisy_images, timesteps, labels, noise = self._prepare_inputs(batch)
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

        if batch_idx == 0:
            self.generate(stage='val')

        return loss

    @torch.no_grad()
    def generate(self, stage='val'):
        num_classes = 5
        num_member_per_class = config.val.num_members_per_class

        samples = torch.randn(size=(num_member_per_class * num_classes, 3, config.image_size, config.image_size), device=self.device)
        labels = torch.tensor([[i] * num_member_per_class for i in range(num_classes)], dtype=torch.int64).flatten().to(self.device)

        self.noise_scheduler.set_timesteps(num_inference_steps=250, device=self.device)

        for i, t in tqdm(enumerate(self.noise_scheduler.timesteps), desc="Generating images from noise"):
            with torch.no_grad():
                noise_pred = self.model(samples, t, labels).to(self.device)

            x = self.noise_scheduler.step(noise_pred, t, samples)
            samples = x.prev_sample.to(self.device)

        samples = samples.detach().cpu().clip(-1, 1)

        folder = config.paths.output.val_images if stage == 'val' else config.paths.output.test_images

        visualize_X_samples_grid(
            samples,
            labels,
            n_samples=num_member_per_class * num_classes,
            n_cols=num_member_per_class,
            filepath=f'{folder}/samples_{self.current_epoch}.png'
        )

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
            dirpath=config.paths.output.checkpoints,
            filename="best_checkpoint",
            save_top_k=1,
            save_last=True,
        )

        progress_bar_callback = TQDMProgressBar(refresh_rate=5)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

        return [checkpoint_callback, early_stop_callback, progress_bar_callback, lr_monitor_callback]
