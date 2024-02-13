from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar

from stereodepth.lit_module import LitStereoDepthEst
from stereodepth.datamodule import KITTI2015DataModule
from stereodepth.models import StereoNet
from stereodepth.losses import MultiScaleLoss


def main():
    # define datamodule
    transforms = A.Compose(
        [
            A.RandomCrop(height=300, width=600),
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image', 'disp': 'mask'}
    )

    datamodule = KITTI2015DataModule(
        data_root='datasets/KITTI2015',
        train_transforms=transforms,
        val_transforms=transforms,
        batch_size=4,
    )

    # define model
    lit_module = LitStereoDepthEst(
        net=StereoNet(),
        loss=MultiScaleLoss(),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename='{epoch}-{train_loss:.3f}',
        save_top_k=1,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress = RichProgressBar()


    # Trainer
    trainer = Trainer(
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        max_epochs=50,
        # accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        # logger=logger,
        accelerator='gpu',
        devices=1,
    )

    # Train Model
    trainer.fit(lit_module, datamodule)


if __name__ == '__main__':
    main()
