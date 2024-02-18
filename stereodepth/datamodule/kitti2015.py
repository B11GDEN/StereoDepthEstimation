import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


class KITTI2015Dataset(Dataset):
    """
    Pytorch dataset for the KITTI2015Dataset.

    Attributes:
        data_root (Path): Root dataset directory
        file_list (list[str]): List of file names with suffix
        transforms (A.Compose | None): Albumentations transformation, Defaults to `None`

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
    ):
        super().__init__()
        self.data_root = data_root
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Returns:
            number of files
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Takes left, right, disp by index, applies augmentations to them.

        Args:
            idx (int): Index of the file

        Returns:
            left (np.ndarray | torch.Tensor): augmented left image.
            right (np.ndarray | torch.Tensor): augmented right image.
            disp (np.ndarray | torch.Tensor): augmented disparity.
            mask (np.ndarray | torch.Tensor): mask with correct disparity pixels.
        """
        left_path = self.data_root / "image_2" / self.file_list[idx]
        right_path = self.data_root / "image_3" / self.file_list[idx]
        disp_path = self.data_root / "disp_occ_0" / self.file_list[idx]

        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256
        mask = (disp < 1) & (disp > 1e-3)

        if self.transforms:
            augment = self.transforms(image=left, right=right, disp=disp, mask=mask)
            # left, right = augment['image'], augment['right']
            # disp, mask = augment['disp'], augment['mask']
            left, right, disp = augment['image'], augment['right'], augment['disp']

        data = {
            "left": left,
            "right": right,
            "disp": disp.unsqueeze(0),
            # "mask": mask.unsqueeze(0),
        }

        return data


class KITTI2015DataModule(LightningDataModule):
    """
    `LightningDataModule` for the KITTI2015 dataset.

    Attributes:
        data_train (Dataset | None): Training dataset
        data_val (Dataset | None): Validation dataset

    Methods:
        __init__: Initialize the `KITTI2015DataModule` object
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
            files = [f.name for f in (data_root / 'training' / 'disp_occ_0').glob('*.png')]

            # split files to train, val
            train_size, val_size = self.hparams.train_val_split
            train_files, val_files = train_test_split(
                files, test_size=val_size, random_state=self.hparams.seed)

            # Datasets
            self.data_train = KITTI2015Dataset(
                data_root=data_root / 'training',
                file_list=train_files,
                transforms=self.hparams.train_transforms,
            )

            self.data_val = KITTI2015Dataset(
                data_root=data_root / 'training',
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


if __name__ == "__main__":
    transforms = A.Compose(
        [
            # A.RandomCrop(height=300, width=600),
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image', 'disp': 'mask'}
    )

    data_root = Path('datasets/KITTI2015/training')
    files = [f.name for f in data_root.glob('image_2/*.png')]
    dataset = KITTI2015Dataset(
        data_root=data_root,
        file_list=files,
        transforms=transforms,
    )

    result = dataset[0]
    disp = (result['disp'] / 192 * 255).cpu().numpy().astype(np.uint8)
    left, right = result['left'], result['right']

    color_disp = cv2.applyColorMap(disp[0], cv2.COLORMAP_JET)

    x = 0
