import cv2
import struct
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


def read_pfm(filename: str) -> np.ndarray:
    """
    Function for read file in pfm format
    """
    with Path(filename).open('rb') as pfm_file:
        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')

        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4

        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale


class Middlebury2014Dataset(Dataset):
    """
    Pytorch dataset for the Middlebury2015Dataset.

    Attributes:
        data_root (Path): Root dataset directory
        file_list (list[str]): List of file names
        transforms (A.Compose | None): Albumentations transformation, Defaults to `None`
        cache_files (bool): cache files to ram memory
        cache (list[list[np.ndarray]]): cache store

    Methods:
        __init__: Initialize the `KITTI2015Dataset` object
        __len__: Return number of files
        __getitem__: Return left_img, right_img and disp by index

    """

    def __init__(
            self,
            data_root: Path,
            file_list: list[str],
            transforms: A.Compose | None = None,
            cache_files: bool = True,
    ):
        super().__init__()
        self.data_root: Path = data_root
        self.file_list: list[str] = file_list
        self.transforms: A.Compose | None = transforms
        self.cache_files: bool = cache_files

        self.cache: list[list[np.ndarray]] = []
        if self.cache_files:
            for f in self.file_list:
                left, right, disp, mask = self.get_data(f)
                self.cache.append([
                    left, right, disp, mask
                ])

    def get_data(self, file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes left, right, disp, mask by index, applies augmentations to them.

        Args:
            file_name (int): file name

        Returns:
            left (np.ndarray | torch.Tensor): left image.
            right (np.ndarray | torch.Tensor): right image.
            disp (np.ndarray | torch.Tensor): disparity.
            mask (np.ndarray | torch.Tensor): mask with correct disparity pixels.
        """
        left_path = self.data_root / file_name / 'im0.png'
        right_path = self.data_root / file_name / 'im1.png'
        disp_path = self.data_root / file_name / 'disp0.pfm'

        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        disp = read_pfm(str(disp_path))
        mask = disp != np.inf
        disp = np.clip(disp, 0, 1)

        return left, right, disp, mask

    def __len__(self) -> int:
        """
        Returns:
            number of files
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Takes left, right, disp, mask by index, applies augmentations to them.

        Args:
            idx (int): Index of the file

        Returns:
            left (np.ndarray | torch.Tensor): augmented left image.
            right (np.ndarray | torch.Tensor): augmented right image.
            disp (np.ndarray | torch.Tensor): augmented disparity.
            mask (np.ndarray | torch.Tensor): mask with correct disparity pixels.
        """
        if self.cache_files:
            left, right, disp, mask = self.cache[idx]
        else:
            left, right, disp, mask = self.get_data(self.file_list[idx])

        if self.transforms:
            augment = self.transforms(image=left, right=right, disp=disp, mask=mask)
            left, right = augment['image'], augment['right']
            disp, mask = augment['disp'], augment['mask']

        data = {
            "left": left,
            "right": right,
            "disp": disp.unsqueeze(0) * 255,
            "mask": mask.unsqueeze(0),
        }

        return data


class Middlebury2014DataModule(LightningDataModule):
    """
    `LightningDataModule` for the Middlebury2014 dataset.

    Attributes:
        data_train (Dataset | None): Training dataset
        data_val (Dataset | None): Validation dataset

    Methods:
        __init__: Initialize the `Middlebury2014DataModule` object
        setup: load data, Split to train val
        train_dataloader: Create and return the train dataloader
        val_dataloader: Create and return the validation dataloader
    """

    def __init__(
            self,
            data_root: str,
            train_transforms: A.Compose | None = None,
            val_transforms: A.Compose | None = None,
            train_val_split: tuple[float, float] = (0.9, 0.1),
            batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 42,
    ) -> None:
        """
        Initialize a `KITTI2015DataModule`.

        Args:
            data_root (str): Path to dataset directory
            train_transforms (A.Compose | None): Albumentations train transformation, Defaults to `None`
            val_transforms (A.Compose | None): Albumentations validation transformation, Defaults to `None`
            train_val_split (tuple[float, float]): The train, validation split.
                Defaults to `(0.9, 0.1)`.
            batch_size (int): The batch size. Defaults to `8`.
            num_workers (int): The number of workers. Defaults to `0`.
            pin_memory (bool): Whether to pin memory. Defaults to `False`.
            seed (int): Random seed. Defaults to `42`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        Args:
            stage (str | None): The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
                Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            data_root = Path(self.hparams.data_root)

            # get all files
            files = [f.name for f in data_root.glob('*')]

            # split files to train, val
            train_size, val_size = self.hparams.train_val_split
            train_files, val_files = train_test_split(
                files, test_size=val_size, random_state=self.hparams.seed)

            # Datasets
            self.data_train = Middlebury2014Dataset(
                data_root=data_root,
                file_list=train_files,
                transforms=self.hparams.train_transforms,
            )

            self.data_val = Middlebury2014Dataset(
                data_root=data_root,
                file_list=val_files,
                transforms=self.hparams.val_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the train dataloader.

        Returns:
            train_dataloader (DataLoader)
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation dataloader.

        Returns:
            val_dataloader (DataLoader)
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


def main():
    transforms = A.Compose(
        [
            # A.RandomCrop(height=512, width=512),
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image', 'disp': 'mask'}
    )

    data_root = Path('datasets/Middlebury2014')
    files = ['Backpack-perfect', 'Bicycle1-perfect']
    dataset = Middlebury2014Dataset(
        data_root=data_root,
        file_list=files,
        transforms=transforms,
    )

    result = dataset[0]
    disp = result['disp'].cpu().numpy().astype(np.uint8)
    left, right = result['left'], result['right']

    color_disp = cv2.applyColorMap(disp[0], cv2.COLORMAP_JET)

    x = 0


if __name__ == '__main__':
    main()
