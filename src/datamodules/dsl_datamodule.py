from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import os
import zarr

class ZarrDataset(Dataset):

    def __init__(self, zarr_file, downsampeling):
        self.zarray = zarr.open(zarr_file, mode="r")
        self.downsampeling = downsampeling
        self.length =  len(self.zarray[self.downsampeling].keys())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.zarray[f"{self.downsampeling}/{idx}/img"][:]
        gt = self.zarray[f"{self.downsampeling}/{idx}/gt"][:]
        return img, gt

class DSLDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        downsampeling: int = 0,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
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


    def setup(self, stage: Optional[str] = None):
        self.data_train = ZarrDataset(os.path.join(self.data_dir, "train.zarr"), self.downsampeling)
        self.data_val = ZarrDataset(os.path.join(self.data_dir, "val.zarr"), self.downsampeling)
        self.data_test = ZarrDataset(os.path.join(self.data_dir, "test.zarr"), self.downsampeling)

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
