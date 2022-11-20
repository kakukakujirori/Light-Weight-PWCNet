"""
Modified from
    albumentations/augmentations/geometric/functional.py
in order to get optical flow.
"""
import random
from typing import Optional, Union
import numpy as np
import cv2

__all__ = [
    "randomPerspective",
    "randomGridDistortion",
]


def get_perspective_mat(
    height: int,
    width: int,
    scale: tuple[float, float] = (0.01, 0.05),
) -> dict:
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

    points[:, 0] *= width
    points[:, 1] *= height

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
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

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
        map_xy[:, :, 0],
        map_xy[:, :, 1],
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=border_val,
    )
    return warped, xy - map_xy


def get_grid_distortion_steps(
    num_steps: int = 5,
    distort_limit: float = 0.3,
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


def randomPerspective(
    imgs: Union[np.ndarray, list[np.ndarray]],
    scale: tuple[float, float] = (0.01, 0.05),
    pad_mode: int = cv2.BORDER_CONSTANT,
    pad_val: int = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[list[np.ndarray], np.ndarray]:

    if isinstance(imgs, np.ndarray):
        mat = get_perspective_mat(*imgs.shape[:2], scale)
        warped, flow = perspective(imgs, mat, border_val=pad_val, border_mode=pad_mode, interpolation=interpolation)
        return warped, flow
    else:
        warped_imgs = []
        for img in imgs:
            mat = get_perspective_mat(*imgs[0].shape[:2], scale)
            warped, flow = perspective(img, mat, border_val=pad_val, border_mode=pad_mode, interpolation=interpolation)
            warped_imgs.append(warped)
        return warped_imgs, flow


def randomGridDistortion(
    imgs: Union[np.ndarray, list[np.ndarray]],
    num_steps: Union[int, tuple[int, int]] = 5,
    distort_limit: float = 0.3,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[int] = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    if isinstance(num_steps, Union[tuple, list]):
        assert len(num_steps) == 2
        assert min(num_steps) >= 2
        num_steps = random.randint(*num_steps)
    xsteps, ysteps = get_grid_distortion_steps(num_steps, distort_limit)

    if isinstance(imgs, np.ndarray):
        warped, flow = grid_distortion(imgs, num_steps, xsteps, ysteps, interpolation=interpolation, border_mode=border_mode, value=value)
        return warped, flow
    else:
        warped_imgs = []
        for img in imgs:
            warped, flow = grid_distortion(img, num_steps, xsteps, ysteps, interpolation=interpolation, border_mode=border_mode, value=value)
            warped_imgs.append(warped)
        return warped_imgs, flow