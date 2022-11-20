import glob
import math
import os
import random
from typing import Optional, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

"""
Modified from
    albumentations/augmentations/geometric/functional.py
in order to get (map_x, map_y).
"""
def get_perspective_mat(
    image: np.ndarray,
    scale: tuple[float, float] = (0.01, 0.05),
) -> dict:
    h, w = image.shape[:2]

    scale = np.random.uniform(*scale)
    points = np.random.normal(0, scale, [4, 2])
    points = np.mod(np.abs(points), 1)

    # top left -- no changes needed, just use jitter
    # top right
    points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
    # bottom right
    points[2] = 1.0 - points[2]  # w = 1.0 - jitt
    # bottom left
    points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

    points[:, 0] *= w
    points[:, 1] *= h

    # Obtain a consistent order of the points and unpack them individually.
    # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
    # here, because the reordered points is used further below.
    def _order_points(pts: np.ndarray) -> np.ndarray:
        pts = np.array(sorted(pts, key=lambda x: x[0]))
        left = pts[:2]  # points with smallest x coordinate - left points
        right = pts[2:]  # points with greatest x coordinate - right points

        if left[0][1] < left[1][1]:
            tl, bl = left
        else:
            bl, tl = left

        if right[0][1] < right[1][1]:
            tr, br = right
        else:
            br, tr = right

        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    points = _order_points(points)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    # do not use width-1 or height-1 here, as for e.g. width=3, height=2
    # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # compute the perspective transform matrix and then apply it
    return cv2.getPerspectiveTransform(points, dst)


def perspective(
    img: np.ndarray,
    matrix: np.ndarray,
    border_val: Union[int, float, list[int], list[float], np.ndarray] = 0,
    border_mode: int = cv2.BORDER_REFLECT_101,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    xy = np.stack([x, y], axis=-1).astype(np.float32)
    map_xy = cv2.perspectiveTransform(xy, np.linalg.inv(matrix))
    
    # warped = cv2.warpPerspective(
    #     img,
    #     M=matrix,
    #     dsize=(w, h),
    #     borderMode=border_mode,
    #     borderValue=border_val,
    #     flags=interpolation,
    # )
    warped = cv2.remap(
        img,
        map_xy[:,:,0],
        map_xy[:,:,1],
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=border_val
    )
    return warped, xy - map_xy


def get_grid_distortion_steps(
    num_steps: int = 5, distort_limit: float = 0.3,
) -> tuple[list[int], list[int]]:
    stepsx = [1 + random.uniform(-distort_limit, distort_limit) for _ in range(num_steps + 1)]
    stepsy = [1 + random.uniform(-distort_limit, distort_limit) for _ in range(num_steps + 1)]
    return stepsx, stepsy


def grid_distortion(
    img: np.ndarray,
    num_steps: int = 5,
    xsteps: tuple = (),
    ysteps: tuple = (),
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Union[int, float, list[int], list[float], np.ndarray, None] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a grid distortion of an input image.
    Reference:
        http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    map_xy = np.stack([map_x, map_y], axis=-1)

    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    xy = np.stack([x, y], axis=-1).astype(np.float32)
    flow = xy - map_xy

    remapped = cv2.remap(
        img,
        map1=map_x,
        map2=map_y,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    return remapped, flow


################################


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
    
    if fg_mask is None:
        mat = get_perspective_mat(img)
        warped_img, map_xy = perspective(img, mat)
        return warped_img, None, map_xy
    else:
        fg_mask = fg_mask.astype(np.uint8)
        # TODO: Try other geometric transformations, should we add an additional Euclid transform?
        num_steps = random.randint(2, 5)
        xsteps, ysteps = get_grid_distortion_steps(num_steps)
        warped_img, map_xy = grid_distortion(img, num_steps, xsteps, ysteps)
        warped_mask, _ = grid_distortion(fg_mask, num_steps, xsteps, ysteps)
        return warped_img, warped_mask.astype(bool), map_xy


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
