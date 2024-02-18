import torch
from torchmetrics import MeanMetric


class EPEMetric(MeanMetric):
    """
    Implementations of EndPoint Error

    Attributes:
        max_disp (float): maximum value of disparity in disparity mask, for KITTI2015 dataset this value is 192
    """
    def __init__(self, max_disp: float = 192):
        super().__init__()
        self.max_disp = max_disp

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        mask = (pred < self.max_disp) & (pred > 1e-3)
        error = torch.abs(target - pred) * mask
        error = torch.flatten(error, 1).sum(-1)
        count = torch.flatten(mask, 1).sum(-1)
        epe = error / count
        super().update(epe[count > 0])