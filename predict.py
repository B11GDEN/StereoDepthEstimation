import cv2
import argparse
import torch
import numpy as np
from stereodepth.models import StereoNet

import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    parser = argparse.ArgumentParser(description="StereoDepth Prediction")

    # input
    parser.add_argument("--left", type=str, default="./imgs/left.png", help="path to left image")
    parser.add_argument("--right", type=str, default="./imgs/right.png", help="path to right image")

    # model
    parser.add_argument("--model", type=str, default="./weights/stereo_net.pt", help="stereo_net weights")

    # output
    parser.add_argument("--output", type=str, default="result_disp.png", help="path to result disparity")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load model
    weights = torch.load(args.model)
    state_dict = {}
    for k, v in weights['state_dict'].items():
        if k.startswith('model.'):
            state_dict[k[6:]] = v

    model = StereoNet()
    model.load_state_dict(state_dict)
    model.eval().cuda()

    # test transform
    transforms = A.Compose(
        [
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        additional_targets={'right': 'image'}
    )

    # input images
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    # inference
    with torch.no_grad():
        augment = transforms(image=left, right=right)
        left, right = augment['image'], augment['right']
        result = model(left.unsqueeze(0).cuda(), right.unsqueeze(0).cuda())

    # transform disparity map
    disp = result['disp'][0, 0].cpu().numpy()
    disp = (disp / 192 * 255).astype(np.uint8)
    color_disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    color_disp = cv2.cvtColor(color_disp, cv2.COLOR_BGR2RGB)

    # save file
    cv2.imwrite(args.output, color_disp)


if __name__ == '__main__':
    main()
