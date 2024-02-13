import torch
import torch.nn as nn
import torch.nn.functional as F


def robust_loss(x, a, c):
    abs_a_sub_2 = abs(a - 2)

    x = x / c
    x = x * x / abs_a_sub_2 + 1
    x = x ** (a / 2)
    x = x - 1
    x = x * abs_a_sub_2 / a
    return x


class MultiScaleLoss(nn.Module):
    def __init__(
            self,
            max_disp: float = 192,
            a: float = 0.8,
            c: float = 0.5,
            A: float = 1,
            tile_size: int = 1,
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.a = a
        self.c = c
        self.A = A
        self.tile_size = tile_size

    def forward(self, preds: dict[str, torch.Tensor], disp: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {}
        for idx, pred in enumerate(preds['multi_scale']):
            scale = disp.size(3) // pred.size(3)
            scale_disp = max(1, scale // self.tile_size)

            disp = disp / scale_disp
            max_disp = self.max_disp / scale_disp

            disp = F.max_pool2d(disp, kernel_size=scale, stride=scale)
            mask = (disp < max_disp) & (disp > 1e-3)
            diff = (pred - disp).abs()

            if self.tile_size > 1 and scale_disp > 1:
                mask = (diff < self.A) & mask

            loss = robust_loss(diff, a=self.a, c=self.c)
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)

            losses[f"disp_loss_{idx}"] = loss
        return losses
