from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from stereodepth.datamodule import KITTI2015DataModule


def main():
    transforms = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image', 'disp': 'mask'}
    )

    datamodule = KITTI2015DataModule(
        data_root='datasets/KITTI2015',
        train_transforms=transforms,
        val_transforms=transforms,
    )

    datamodule.setup()
    loader = datamodule.train_dataloader()

    for batch in loader:
        x = 0


if __name__ == '__main__':
    main()
