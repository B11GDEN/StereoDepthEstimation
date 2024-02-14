import cv2
import torch
import numpy as np
from stereodepth.models import StereoNet
from stereodepth.datamodule import KITTI2015DataModule

import albumentations as A
from albumentations.pytorch import ToTensorV2


def main():
    weights = torch.load('sber-task/v3mmznl0/checkpoints/epoch=73-train_loss=0.763.ckpt')
    state_dict = {}
    for k, v in weights['state_dict'].items():
        if k.startswith('net'):
            state_dict[k[4:]] = v

    model = StereoNet()
    model.load_state_dict(state_dict)
    model.eval().cuda()

    transforms = A.Compose(
        [
            # A.RandomCrop(height=300, width=600),
            A.Normalize(),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image', 'disp': 'mask'}
    )

    datamodule = KITTI2015DataModule(
        data_root='datasets/KITTI2015',
        train_transforms=transforms,
        val_transforms=transforms,
        batch_size=1,
    )

    datamodule.setup()

    # with torch.no_grad():
    for batch in datamodule.train_dataloader():
        left, right = batch['left'], batch['right']
        result = model(left.cuda(), right.cuda())
        disp = (result['disp'][0, 0] / 192 * 255).detach().cpu().numpy().astype(np.uint8)
        color_disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        x = 0


if __name__ == '__main__':
    main()
