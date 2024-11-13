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
            N = 2_000
            batch_size = images.shape[0]
            indices = torch.randperm(batch_size)[:N]
            images = images[indices]
            labels = labels[indices]

            from sklearn.model_selection import StratifiedShuffleSplit
            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.train.seed)
            train_idx, val_idx = next(iter(stratified_split.split(images, labels)))
            train_images, val_images = images[train_idx], images[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # Check unique labels and count of each class in each set
            unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
            unique_val_labels, val_counts = np.unique(val_labels, return_counts=True)

            logger.info(f'Unique labels in training set: {unique_train_labels}')
            logger.info(f'Counts of each class in training set: {dict(zip(unique_train_labels, train_counts))}')

            logger.info(f'Unique labels in validation set: {unique_val_labels}')
            logger.info(f'Counts of each class in validation set: {dict(zip(unique_val_labels, val_counts))}')

            assert len(unique_train_labels) == 5, "Training set does not contain all 5 classes"
            assert len(unique_val_labels) == 5, "Validation set does not contain all 5 classes"

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
