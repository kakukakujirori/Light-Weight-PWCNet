import glob
import os
import random
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class AutoFlowDataset(Dataset):
    """AutoFlow Dataset for optical flow estimation task.

    Params:
        data_root (str): The directory in which video data are stored
        videolist (str): The file which contains pairs of videoname and its frame length
        max_frame_diff (int): Maximally how many frames can be skipped for computing flow (default: 4)
        width (int): The width of the final image (default: 512)
        height (int): The height of the final image (default: 512)
        is_train (bool): If True, augmentations are applied
    Return:
        two image tensors [C, H, W], [C, H, W]
    """

    def __init__(
        self,
        image_dir: str,
        layer_num: int = 3,
        width: int = 512,
        height: int = 512,
        gaussian_blur_prob: float = 0.2,
        motion_blur_prob: float = 0.2,
        fog_prob: float = 0.05,
    ):
        assert layer_num > 1, f"{layer_num=} must be >= 2 (at least one background and foreground)"
        assert 0 <= gaussian_blur_prob <= 1
        assert 0 <= motion_blur_prob <= 1
        assert 0 <= fog_prob <= 1
        self.image_dir = image_dir
        self.layer_num = layer_num
        self.width = width
        self.height = height
        self.gaussian_blur_prob = gaussian_blur_prob
        self.motion_blur_prob = motion_blur_prob
        self.fog_prob = fog_prob

        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(
            os.path.join(image_dir, "*.png")
        )

        self.resize_transforms = A.Compose(
            [
                A.HorizontalFlip(),
                A.RandomResizedCrop(height, width, scale=(0.75, 1), always_apply=True),
            ]
        )
        self.augmentations = A.Compose(
            [
                A.RandomBrightnessContrast(),
                A.ImageCompression(),
                A.GaussNoise(),
            ]
        )
        self.randomFog = A.Compose(
            [
                A.RandomFog(p=fog_prob),
            ],
            additional_targets={"warped": "image"},
        )
        self.totensor = A.Compose(
            [
                A.ToFloat(always_apply=True),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bg = cv2.imread(self.image_paths[index])
        fg_paths = random.sample(self.image_paths, self.layer_num - 1)
        fgs = [cv2.imread(fg_path) for fg_path in fg_paths]
        assert bg is not None, f"Failed to load {self.image_paths[index]}"
        for fg, fg_p in zip(fgs, fg_paths):
            assert fg is not None, f"Failed to load {fg_p}"

        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        fgs = [cv2.cvtColor(fg, cv2.COLOR_BGR2RGB) for fg in fgs]
        bg = self.resize_transforms(image=bg)["image"]
        fgs = [self.resize_transforms(image=fg)["image"] for fg in fgs]

        # 3.1: object masks
        fg_masks = [
            create_polygon_mask(self.height, self.width, (self.height + self.width) // 4)
            for _ in fgs
        ]

        # 3.2: motion model
        warped_bg, _, bg_flow = warp(bg, None)
        warped_fgs, warped_fg_masks, fg_forward_flows = zip(
            *[warp(fg, fg_mask) for fg, fg_mask in zip(fgs, fg_masks)]
        )
        fg_backward_flows = [-flw for flw in fg_forward_flows]

        # 3.3: visual effects (here compose images)
        applied_gaussian_blur_ksize = [
            random.choice([3, 5, 7]) if random.random() < self.gaussian_blur_prob else 0
            for _ in [bg] + fgs
        ]
        applied_motion_blur_ksize = [
            random.choice([3, 5, 7]) if random.random() < self.motion_blur_prob else 0
            for _ in [bg] + fgs
        ]

        original = compose(
            bg,
            bg_flow,
            fgs,
            fg_masks,
            fg_forward_flows,
            applied_gaussian_blur_ksize,
            applied_motion_blur_ksize,
        )
        warped = compose(
            warped_bg,
            -bg_flow,
            warped_fgs,
            warped_fg_masks,
            fg_backward_flows,
            applied_gaussian_blur_ksize,
            applied_motion_blur_ksize,
        )
        flow = compose(
            bg_flow,
            bg_flow,
            fg_forward_flows,
            fg_masks,
            fg_forward_flows,
            [0] * (1 + len(fgs)),
            [0] * (1 + len(fgs)),
        )

        # augmentation & preprocess
        fog_applied = self.randomFog(image=original, warped=warped)
        original, warped = fog_applied["image"], fog_applied["warped"]
        original = self.augmentations(image=original)["image"]
        warped = self.augmentations(image=warped)["image"]
        original = self.totensor(image=original)["image"]
        warped = self.totensor(image=warped)["image"]
        flow = self.totensor(image=flow)["image"]

        return original, warped, flow


def create_polygon_mask(
    height: int = 512, width: int = 512, polygon_size: int = 128
) -> np.ndarray:
    n = random.randint(3, 10)
    dists_from_center = [random.uniform(polygon_size / 5, polygon_size) for _ in range(n)]
    angles = sorted(x * np.pi / 180 for x in random.sample(range(320), n))

    init_pt = np.array([random.uniform(0, height), random.uniform(0, width)])
    points = [
        init_pt + d * np.array([np.cos(theta), np.sin(theta)])
        for d, theta in zip(dists_from_center, angles)
    ]
    points = np.stack([pt.astype(np.int64) for pt in points])

    mask = cv2.fillPoly(
        np.zeros((height, width), dtype=np.uint8),
        pts=[points],
        color=1,
    )

    # hole
    if mask.sum() > width * height // 32:
        n2 = random.randint(3, 10)
        polygon_size2 = min(dists_from_center)
        dists_from_center2 = [random.uniform(polygon_size2 / 5, polygon_size2) for _ in range(n2)]
        angles2 = sorted(x * np.pi / 180 for x in random.sample(range(320), n2))

        points = [
            init_pt + d * np.array([np.cos(theta), np.sin(theta)])
            for d, theta in zip(dists_from_center2, angles2)
        ]
        points = np.stack([pt.astype(np.int64) for pt in points])

        mask += cv2.fillPoly(
            np.zeros((height, width), dtype=np.uint8),
            pts=[points],
            color=1,
        )
        mask %= 2

    return mask.astype(bool)


def warp(
    img: np.ndarray, fg_mask: Optional[np.ndarray]
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if fg_mask is not None:
        assert (
            img.shape[:2] == fg_mask.shape
        ), f"Inconsistent shapes: {img.shape=}, {fg_mask.shape=}"
    height, width, _ = img.shape
    coord_x, coord_y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    coord = np.stack([coord_x, coord_y], axis=-1).astype(np.float32)

    if fg_mask is None:
        aug = A.Compose(
            [
                A.Perspective(pad_mode=cv2.BORDER_REFLECT_101, always_apply=True),
            ],
            additional_targets={"coord": "image"},
        )
        warped = aug(image=img, coord=coord)
        return warped["image"], None, warped["coord"] - coord
    else:
        fg_mask = fg_mask.astype(np.uint8)
        # TODO: Try other geometric transformations, should we add an additional Euclid transform?
        aug = A.Compose(
            [
                A.GridDistortion(
                    num_steps=random.randint(2, 5),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    always_apply=True,
                ),
            ],
            additional_targets={"mask": "image", "coord": "image"},
        )
        warped = aug(image=img, mask=fg_mask, coord=coord)
        return (
            warped["image"],
            warped["mask"].astype(bool),
            coord - warped["coord"],  # this is the forward-warp
        )


def compose(
    bg: np.ndarray,
    bg_flow: np.ndarray,
    fgs: list[np.ndarray],
    fg_masks: list[np.ndarray],
    fg_flows: list[np.ndarray],
    applied_gaussian_blur_ksize: list[int],
    applied_motion_blur_ksize: list[int],
) -> np.ndarray:
    assert (
        len(fgs)
        == len(fg_masks)
        == len(fg_flows)
        == len(applied_gaussian_blur_ksize) - 1
        == len(applied_motion_blur_ksize) - 1
    )

    def _motion_blur(image: np.ndarray, size: int, angle: float) -> np.ndarray:
        assert size % 2 > 0
        k = np.zeros((size, size), dtype=np.float32)
        k[size // 2, :] = 1
        k = cv2.warpAffine(
            k, cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0), (size, size)
        )
        k /= np.sum(k)
        return cv2.filter2D(image, -1, k)

    # background
    ret = bg.copy()
    if applied_gaussian_blur_ksize[0] >= 3:
        ksize = applied_gaussian_blur_ksize[0]
        ret = cv2.GaussianBlur(ret, (ksize, ksize), sigmaX=0)
    if applied_motion_blur_ksize[0] >= 3:
        flow_sum = np.sum(bg_flow, axis=(0, 1))
        flow_angle = np.arctan2(flow_sum[1], flow_sum[0]) * 180 / np.pi
        ret = _motion_blur(ret, applied_motion_blur_ksize[0], flow_angle)

    # foregrounds
    for fg, msk, flow, gaussian_blur_ksize, motion_blur_ksize in zip(
        fgs, fg_masks, fg_flows, applied_gaussian_blur_ksize[1:], applied_motion_blur_ksize[1:]
    ):

        msk = msk.astype(np.float32)
        # blur augmentations
        if gaussian_blur_ksize >= 3:
            assert gaussian_blur_ksize % 2 == 1
            fg = cv2.GaussianBlur(fg, (gaussian_blur_ksize, gaussian_blur_ksize), sigmaX=0)
            msk = cv2.GaussianBlur(msk, (gaussian_blur_ksize, gaussian_blur_ksize), sigmaX=0)
        if motion_blur_ksize >= 3:
            flow_sum = np.sum(flow * msk[:, :, None], axis=(0, 1))
            flow_angle = np.arctan2(flow_sum[1], flow_sum[0]) * 180 / np.pi
            fg = _motion_blur(fg, motion_blur_ksize, flow_angle)
            msk = _motion_blur(msk, motion_blur_ksize, flow_angle)
        # compose
        ret = msk[:, :, None] * fg + (1 - msk[:, :, None]) * ret

    if bg.dtype == np.uint8:
        ret = np.clip(ret, 0, 255)

    return ret.astype(bg.dtype)
