import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning import LightningModule
from torchmetrics import Metric, MeanMetric


class LitStereoDepthEst(LightningModule):
    """
    `LightningModule` for stereo depth estimation task.
    """

    def __init__(
            self,
            net: torch.nn.Module,
            loss: torch.nn.Module,
            learning_rate: float = 3e-4,
    ) -> None:
        """
        Initialize a `LitStereoDepthEst`.

        Args:
            net (torch.nn.Module):
            loss (torch.nn.Module):
            learning_rate (float): Initial learning rate, Defaults to `3e-4`
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net', 'loss', 'learning_rate'])

        self.net: torch.nn.Module = net
        self.loss: torch.nn.Module = loss

        # Need this variable for learning rate tuner
        self.learning_rate: float = learning_rate

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            left (torch.Tensor): input left tensor
            right (torch.Tensor): input right tensor
        Returns:
            output (torch.Tensor): output tensor
        """
        return self.net(left, right)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step. Compute prediction, loss and log them.

        Args:
            batch (dict[str, torch.Tensor]): Batch output from dataloader.
            batch_idx (int): number of batch

        Returns:
            loss (torch.Tensor): calculated loss
        """
        left, right, disp = batch['left'], batch['right'], batch['disp']

        preds = self.forward(left, right)
        losses = self.loss(preds, disp)
        train_loss = sum(losses.values()) / len(losses)

        self.log_dict(losses, prog_bar=True)
        self.log('train_loss', train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Single validation step. Compute prediction, loss and log them.

        Args:
            batch (dict[str, torch.Tensor]): Batch output from dataloader.
            batch_idx (int): number of batch
        """
        # left, right, disp = batch['left'], batch['right'], batch['disp']
        #
        # preds = self.forward(left, right)
        # loss = self.loss(preds, disp)
        #
        # self.log_dict(loss, prog_bar=True)
        pass

    def configure_optimizers(self):
        """
        Set optimizer and sheduler
        """
        optimizer = AdamW(params=self.trainer.model.parameters(), lr=self.learning_rate)
        sheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [sheduler]


if __name__ == '__main__':
    x = 0
