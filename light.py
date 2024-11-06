from utils import logger, make_clear_directory
from config import config
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.utils import make_image_grid
from diffusers.optimization import get_cosine_schedule_with_warmup

torch.set_float32_matmul_precision('medium')


class SpriteLightning(pl.LightningModule):
    def __init__(self, train_dl_len):
        super().__init__()
        self.train_dl_len = train_dl_len

        from model import SpriteModel
        self.model = SpriteModel()

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.save_hyperparameters(ignore=['model', 'noise_scheduler'])

        make_clear_directory(config.dirs.output_val_images)
        make_clear_directory(config.dirs.output_test_images)

    def forward(self, noisy_images, labels, timesteps):
        x = self.model(noisy_images, labels, timesteps)
        return x

    def shared_step(self, batch):
        clean_images, labels = batch
        noise = torch.randn(clean_images.shape)
        batch_size = clean_images.shape[0]

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            dtype=torch.int64
        )

        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict noise residual
        noise_pred = self.model(noisy_images, labels, timesteps)
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

    def on_epoch_end(self):
        if self.global_rank != 0:  # Only save from the main process
            return

        if self.current_epoch % config.train.save_image_epochs == 0 or self.current_epoch == config.train.max_epochs - 1:
            self.evaluate(self.current_epoch)

    def evaluate(self, epoch):
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)

        images = pipeline(
            batch_size=config.train.val_batch_size,
            generator=torch.Generator(device='cpu').manual_seed(config.train.seed),
        ).images

        image_grid = make_image_grid(images, rows=4, cols=4)

        # Save the images
        image_grid.save(f"{config.dirs.output_val_images}/{epoch:04d}.png")

    def generate(self, count=16):
        pipeline = DDPMPipeline(unet=self.model, scheduler=self.noise_scheduler)

        images = pipeline(
            batch_size=count,
            generator=torch.Generator(device='cpu'),
        ).images

        image_grid = make_image_grid(images, rows=4, cols=4)

        # Save the images

        from datetime import datetime
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        image_grid.save(f"{config.dirs.output_test_images}/{date_time_str}.png")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.train.learning_rate)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(self.train_dl_len * config.train.max_epochs),
        )

        # lr_scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         optimizer,
        #         mode='min',
        #         patience=2,
        #         factor=0.5
        #     ),
        #     "monitor": "train_loss",
        #     "interval": "epoch",
        #     "frequency": 1,
        # }

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
