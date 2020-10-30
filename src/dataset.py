from typing import Callable, Any
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = True
        self.dims = (1, 28, 28)

    def prepare_data(self) -> None:
        # download
        MNIST(self.data_dir, train=True, download=False)
        MNIST(self.data_dir, train=False, download=False)

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(
                self.data_dir, train=True, transform=transforms.ToTensor()
            )
            self.train_dataset, self.val_dataset = random_split(
                mnist_full, [55000, 5000]
            )
        if stage == 'test' or stage is None:
            self.test_dataset = MNIST(
                self.data_dir, train=False, transform=transforms.ToTensor()
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    datamodule = MNISTDataModule('/dataset/MNIST', 1, None, None)
    print(datamodule)
    datamodule.prepare_data()
