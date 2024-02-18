import argparse

import albumentations as A
from albumentations.pytorch import ToTensorV2

from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar

from stereodepth.lit_module import LitStereoDepthEst
from stereodepth.datamodule import KITTI2015DataModule, Middlebury2014DataModule
from stereodepth.models import StereoNet
from stereodepth.losses import MultiScaleLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Training Cycle for StereoDepth Estimation Task")

    # dataloader
    parser.add_argument("--batch-size", type=int, default=8, help="training batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="nember of workers in dataloader")
    parser.add_argument("--pin-memory", action="store_true", help="pin gpu memory in dataloader")

    # training
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="initial learning rate")
    parser.add_argument("--crop-size", type=int, nargs="+", default=(300, 600),  help="training crop size")
    parser.add_argument("--acc-grad-batches", type=int, default=1, help="number of batches to accumulate during training")
    parser.add_argument("--max-epochs", type=int, default=100, help="maximum number of epochs")
    parser.add_argument("--check-val", type=int, default=1, help="check val every n epochs")

    # device
    parser.add_argument("--accelerator", type=str, default="gpu", help="type of accelerator, (gpu, cpu)")
    parser.add_argument("--devices", type=int, nargs="+", default=1, help="indexes of devices")

    # wandb
    parser.add_argument("--project", type=str, default='sber-task', help="name of project")
    parser.add_argument("--exp", type=str, default='kitti2015', help="name of experiment")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # define datamodule
    train_transforms = A.Compose(
        [
            A.RandomCrop(height=args.crop_size[0], width=args.crop_size[1]),
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image', 'disp': 'mask'}
    )
    val_transforms = A.Compose(
        [
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image', 'disp': 'mask'}
    )

    datamodule = KITTI2015DataModule(
        data_root='datasets/KITTI2015',
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # define model
    lit_module = LitStereoDepthEst(
        model=StereoNet(),
        loss=MultiScaleLoss(),
        learning_rate=args.learning_rate
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_epe',
        filename='{epoch}-{val_epe:.3f}',
        save_top_k=1,
        mode='min',
        # save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress = RichProgressBar()

    # Logger
    logger = WandbLogger(
        project=args.project,
        name=args.exp,
    )

    # Trainer
    trainer = Trainer(
        check_val_every_n_epoch=args.check_val,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.acc_grad_batches,
        callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        logger=logger,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    # Find best learning rate
    # tuner = Tuner(trainer)
    # tuner.lr_find(lit_module, datamodule,
    #               max_lr=1e-2, min_lr=1e-8, num_training=1000)

    # Train Model
    trainer.fit(lit_module, datamodule)


if __name__ == '__main__':
    main()
