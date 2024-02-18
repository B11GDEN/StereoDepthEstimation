import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning import LightningModule
from torchmetrics import Metric, MetricCollection
from stereodepth.metrics import EPEMetric, RateMetric


class LitStereoDepthEst(LightningModule):
    """
    `LightningModule` for stereo depth estimation task.

    Implements training and validation logic.

    Attributes:
        model (torch.nn.Module): network for training, e.g. StereoNet
        loss (torch.nn.Module): training loss, e.g. MultiScaleLoss
        learning_rate (float): Initial learning rate, this attribute is also needed because of
            Lightning Tuner to tune the value
        train_metric (MetricCollection): train metrics to log in wandb
        val_metric (MetricCollection): val metrics to log in wandb

    Methods:
        __init__: Initialize the `LitSegmenter` object
        forward: Forward pass
        training_step: Single training step on a batch of data
        on_train_epoch_end: Lightning hook that is called when train epoch ends
        validation_step: Single validation step on a batch of data
        on_validation_epoch_end: Lightning hook that is called when validation epoch ends
        configure_optimizers: Сonfigure optimizer nad sheduler
    """

    def __init__(
            self,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            learning_rate: float = 3e-4,
    ) -> None:
        """
        Initialize a `LitStereoDepthEst`.

        Args:
            model (torch.nn.Module): model to perform stereo depth estimation task
            loss (torch.nn.Module): loss function
            learning_rate (float): Initial learning rate, Defaults to `3e-4`
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['model', 'loss', 'learning_rate'])

        self.model: torch.nn.Module = model
        self.loss: torch.nn.Module = loss

        # Need this variable for learning rate tuner
        self.learning_rate: float = learning_rate

        # train and validation metrics
        metric = MetricCollection(
            {
                "epe": EPEMetric(),
                "rate_1": RateMetric(1.0),
                "rate_3": RateMetric(3.0),
            }
        )
        self.train_metric: MetricCollection = metric.clone(prefix="train_")
        self.val_metric: MetricCollection = metric.clone(prefix="val_")

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            left (torch.Tensor): input left tensor
            right (torch.Tensor): input right tensor
        Returns:
            output (torch.Tensor): output tensor
        """
        return self.model(left, right)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step. Compute prediction, loss, metrics and log them.

        Args:
            batch (dict[str, torch.Tensor]): Batch output from dataloader.
            batch_idx (int): number of batch

        Returns:
            loss (torch.Tensor): calculated loss
        """

        # forward
        preds = self.forward(batch['left'], batch['right'])

        # calculate loss
        losses = self.loss(preds, batch['disp'])
        train_loss = sum(losses.values()) / len(losses)

        # log loss
        self.log_dict(losses, prog_bar=True)
        self.log('train_loss', train_loss, prog_bar=True)

        # calculate metrics
        self.train_metric(preds['disp'], batch['disp'])

        return train_loss

    def on_train_epoch_end(self) -> None:
        """
        Log train metrics and reset
        """
        self.log_dict(self.train_metric.compute())
        self.train_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Single validation step. Compute prediction, metrics and log them.

        Args:
            batch (dict[str, torch.Tensor]): Batch output from dataloader.
            batch_idx (int): number of batch
        """

        # forward
        preds = self.forward(batch['left'], batch['right'])

        # calculate metrics
        self.val_metric(preds['disp'], batch['disp'])

    def on_validation_epoch_end(self) -> None:
        """
        Log validation metrics and reset
        """
        self.log_dict(self.val_metric.compute())
        self.val_metric.reset()

    def configure_optimizers(self):
        """
        Set optimizer and sheduler
        """
        optimizer = AdamW(params=self.trainer.model.parameters(), lr=self.learning_rate)
        sheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [sheduler]
