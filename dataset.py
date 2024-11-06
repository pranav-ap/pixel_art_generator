from utils import logger
from config import config
import os
import torch
import numpy as np
import lightning as L
from typing import Optional
from torchvision import transforms as T


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

        self.num_workers = os.cpu_count()

        self.transform = T.Compose(
            [
                T.ConvertImageDtype(torch.float),
                T.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ]
        )

        self.train_dataset: Optional[SpriteDataset] = None
        self.val_dataset: Optional[SpriteDataset] = None

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            filepath = f'{config.dirs.data}/sprites.npy'
            assert os.path.exists(filepath)
            images = np.load(filepath)
            logger.debug(f'images.shape {images.shape}')
            images = images.reshape(-1, 3, 16, 16)
            
            filepath = f'{config.dirs.data}/sprites_labels.npy'
            assert os.path.exists(filepath)
            labels = np.load(filepath)

            from sklearn.model_selection import train_test_split
            train_images, val_images, train_labels, val_labels = train_test_split(
                images, labels, test_size=0.2, random_state=42, shuffle=True
            )

            self.train_dataset = SpriteDataset(
                images=torch.tensor(train_images),
                labels=torch.tensor(train_labels),
                subset="train",
                transform=self.transform
            )

            self.val_dataset = SpriteDataset(
                images=torch.tensor(val_images),
                labels=torch.tensor(val_labels),
                subset="validate",
                transform=self.transform
            )

            logger.info(f"Train Dataset       : {len(self.train_dataset)} samples")
            logger.info(f"Validation Dataset  : {len(self.val_dataset)} samples")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.train.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.train.val_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
