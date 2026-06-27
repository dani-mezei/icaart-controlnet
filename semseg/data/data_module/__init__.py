import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torchvision import transforms

from semseg.data.dataset.combined_dataset import CombinedDataset
from semseg.data.dataset.kitti_dataset import KittiDataset
from semseg.data.sampler import MixedBatchSampler
from semseg.utils import seed_worker


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, label):
        # Convert numpy array to PIL Image
        label_pil = Image.fromarray(label.astype(np.uint8))
        # Resize using nearest neighbor to preserve class values
        label_resized = label_pil.resize(
            size=(self.size[1], self.size[0]), resample=Image.NEAREST
        )
        return np.array(label_resized)


class MixedDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(self.args.height, self.args.width)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(self.args.height, self.args.width)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        label_transform = transforms.Compose(
            [Resize(size=(self.args.height, self.args.width))]
        )

        if self.args.use_synthetic_images:
            self.real_train_dataset = KittiDataset(
                self.args.train_image_dir,
                self.args.train_label_dir,
                image_transform=train_transform,
                label_transform=label_transform,
                additional_image_dirs=(
                    self.args.additional_train_image_dir
                    if self.args.additional_train_image_dir
                    else None
                ),
                additional_label_dirs=(
                    self.args.additional_train_label_dir
                    if self.args.additional_train_label_dir
                    else None
                ),
                negative_image_dirs=(
                    self.args.negative_train_image_dir
                    if self.args.negative_train_image_dir
                    else None
                ),
                negative_label_dirs=(
                    self.args.negative_train_label_dir
                    if self.args.negative_train_label_dir
                    else None
                ),
                max_samples=(
                    self.args.max_real_samples if self.args.max_real_samples else None
                ),
            )

            self.synthetic_train_dataset = KittiDataset(
                self.args.synthetic_image_dir,
                self.args.synthetic_label_dir,
                image_transform=train_transform,
                label_transform=label_transform,
                max_samples=(
                    self.args.max_synth_samples if self.args.max_synth_samples else None
                ),
            )

            self.combined_train_dataset = CombinedDataset(
                self.real_train_dataset, self.synthetic_train_dataset
            )

        else:
            self.train_dataset = KittiDataset(
                self.args.train_image_dir,
                self.args.train_label_dir,
                image_transform=train_transform,
                label_transform=label_transform,
                additional_image_dirs=(
                    self.args.additional_train_image_dir
                    if self.args.additional_train_image_dir
                    else None
                ),
                additional_label_dirs=(
                    self.args.additional_train_label_dir
                    if self.args.additional_train_label_dir
                    else None
                ),
                negative_image_dirs=(
                    self.args.negative_train_image_dir
                    if self.args.negative_train_image_dir
                    else None
                ),
                negative_label_dirs=(
                    self.args.negative_train_label_dir
                    if self.args.negative_train_label_dir
                    else None
                ),
                max_samples=(
                    self.args.max_real_samples if self.args.max_real_samples else None
                ),
            )

        self.val_dataset = KittiDataset(
            self.args.val_image_dir,
            self.args.val_label_dir,
            image_transform=val_transform,
            label_transform=label_transform,
            additional_image_dirs=(
                self.args.additional_val_image_dir
                if self.args.additional_val_image_dir
                else None
            ),
            additional_label_dirs=(
                self.args.additional_val_label_dir
                if self.args.additional_val_label_dir
                else None
            )
        )

    def train_dataloader(self):
        if self.args.use_synthetic_images:
            if self.args.mixed_batch:

                if self.args.distributed:
                    base_sampler = DistributedSampler(self.combined_train_dataset)
                else:
                    base_sampler = SequentialSampler(self.combined_train_dataset)

                batch_sampler = MixedBatchSampler(
                    sampler=base_sampler,
                    batch_size=self.args.batch_size,
                    drop_last=True,
                    real_size=len(self.real_train_dataset),
                    synthetic_size=len(self.synthetic_train_dataset),
                    real_ratio=self.args.real_ratio,
                    shuffle=True,
                )

                return DataLoader(
                    self.combined_train_dataset,
                    batch_sampler=batch_sampler,
                    num_workers=3,
                    persistent_workers=True,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(42),
                )

            else:
                return DataLoader(
                    self.combined_train_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=3,
                    persistent_workers=True,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(42),
                )

        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=3,
                persistent_workers=True,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(42),
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=3,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(42),
        )
