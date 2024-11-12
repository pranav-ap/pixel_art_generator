import os
from typing import Optional

import lightning as L
import numpy as np
import torch
from torchvision import transforms as T

from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class SpriteDataset(torch.utils.data.Dataset):
    # noinspection PyTypeChecker
    def __init__(self, images, labels, subset, transform=None):
        self.subset = subset
        self.transform = transform

        assert len(images) == len(labels)

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SpriteDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = os.cpu_count()  # 1 3 os.cpu_count()

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
            ]
        )

        self.train_dataset: Optional[SpriteDataset] = None
        self.val_dataset: Optional[SpriteDataset] = None

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            filepath = f'{config.dirs.data}/sprites.npy'
            logger.info('Loading Sprites')
            assert os.path.exists(filepath), f'{filepath} not found'
            images = np.load(filepath)
            images = images.reshape(-1, 3, 16, 16)
            # Constrain between 0 and 1
            images = images / 255.

            filepath = f'{config.dirs.data}/sprites_labels.npy'
            logger.info('Loading Sprite Labels')
            assert os.path.exists(filepath), f'{filepath} not found'
            labels = np.load(filepath)
            # Convert one-hot to number
            labels = labels.argmax(axis=1)

            # Choose N samples
            N = 1_00
            batch_size = images.shape[0]
            indices = torch.randperm(batch_size)[:N]
            images = images[indices]
            labels = labels[indices]

            from sklearn.model_selection import train_test_split
            train_images, val_images, train_labels, val_labels = train_test_split(
                images, labels, test_size=0.2, random_state=42, shuffle=True
            )

            self.train_dataset = SpriteDataset(
                images=torch.tensor(train_images, dtype=torch.float32),
                labels=torch.tensor(train_labels, dtype=torch.int32),
                subset="train",
            )

            self.val_dataset = SpriteDataset(
                images=torch.tensor(val_images, dtype=torch.float32),
                labels=torch.tensor(val_labels, dtype=torch.int32),
                subset="validate",
            )

            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
