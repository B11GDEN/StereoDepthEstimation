import torch
import torch.nn as nn
import torch.nn.functional as F


def robust_loss(x: torch.Tensor, a: float, c: float) -> torch.Tensor:
    """
    Robust Loss Function. From paper: A General and Adaptive Robust Loss Function

    Args:
        x (torch.Tensor): input Tensor
        a (float): shape parameter that controls the robustness of the loss
        c (float): a scale parameter that controls the size of the loss’s quadratic
            bowl near x = 0.
    """
    abs_a_sub_2 = abs(a - 2)

    x = x / c
    x = x * x / abs_a_sub_2 + 1
    x = x ** (a / 2)
    x = x - 1
    x = x * abs_a_sub_2 / a
    return x


class MultiScaleLoss(nn.Module):
    """
    Multi Scale Loss Function for Stereo-Depth Estimation Task

    Attributes:
        max_disp (float): maximum value of disparity in disparity mask, for KITTI2015 dataset this value is 192
        a (float): shape parameter of robust loss
        c (float): scale parameter of robust loss
    """
    def __init__(
            self,
            max_disp: float = 192,
            a: float = 0.8,
            c: float = 0.5,
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.a = a
        self.c = c

    def forward(self, preds: dict[str, torch.Tensor], disp: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {}
        for idx, pred in enumerate(preds['multi_scale']):
            scale = disp.size(3) // pred.size(3)

            disp = F.max_pool2d(disp, kernel_size=scale, stride=scale)
            mask = (disp < self.max_disp) & (disp > 1e-3)
            diff = (pred - disp).abs()

            loss = robust_loss(diff, a=self.a, c=self.c)
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)

            losses[f"disp_loss_{idx}"] = loss
        return losses
