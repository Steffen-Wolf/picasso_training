from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os
import zarr

class ZarrDataset(Dataset):

    def __init__(self, zarr_file, downsampeling, channels=None):
        self.zarray = zarr.open(zarr_file, mode="r")
        self.downsampeling = downsampeling
        self.length =  len(self.zarray[self.downsampeling].keys())
        mask_dict = {
            'ugriz': None,
            'ugri' : [4],
            'gri'  : [0,4],
            'gr'   : [0,3,4],
            'r'    : [0,1,3,4],
        }
        self.channel_mask = mask_dict[channels]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.zarray[f"{self.downsampeling}/{idx}/img"][:]
        if self.channel_mask is not None:
            img[self.channel_mask] = 0.
        gt = self.zarray[f"{self.downsampeling}/{idx}/gt"][:]
        return img, gt

class DSLDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        downsampeling: int = 0,
        dataset_size = 0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        channels: str = "ugriz"
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.downsampeling = downsampeling

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset_size = dataset_size
        self.channels = channels

    def setup(self, stage: Optional[str] = None):
        self.data_train = ZarrDataset(zarr_file = os.path.join(self.data_dir, "train.zarr"),
                                      downsampeling = self.downsampeling,
                                      channels = self.channels)
        if self.dataset_size > 0:
            l = len(self.data_train)
            indices = torch.randperm(l)[:self.dataset_size].repeat(l // self.dataset_size)
            print("reducing dataset to size", self.dataset_size, "repeating to ", len(indices), "~", l)
            self.data_train = Subset(self.data_train, indices)

        self.data_val = ZarrDataset(zarr_file = os.path.join(self.data_dir, "val.zarr"),
                                    downsampeling = self.downsampeling,
                                    channels = self.channels)
        self.data_test = ZarrDataset(zarr_file = os.path.join(self.data_dir, "test.zarr"),
                                     downsampeling = self.downsampeling,
                                     channels = self.channels)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
