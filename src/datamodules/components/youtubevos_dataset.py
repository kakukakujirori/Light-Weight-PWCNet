import os, io, random
import json
import zipfile
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset


class ZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = ZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, 'r')
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = ZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        im = Image.open(io.BytesIO(data))
        return im


################################################################


class YouTubeVOSDataset(Dataset):
    """
    Dataset for video depth estimation task.
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
    def __init__(self,
                 data_root: str,
                 videolist: str,
                 max_frame_diff: int = 4,
                 width: int = 512,
                 height: int = 512,
                 is_train: bool = True):
        self.data_root = data_root
        self.max_frame_diff = max_frame_diff
        self.is_train = is_train

        json_path = os.path.join(videolist)
        with open(json_path, 'r') as f:
            self.video_dict = json.load(f)  # {video_name: max_frame, ...}
        self.video_names = list(self.video_dict.keys())

        self.train_transforms = A.ReplayCompose([
            # augmentations
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.ImageCompression(),
            # preprocess
            A.RandomResizedCrop(height, width, scale=(0.75, 1), always_apply=True),
            A.ToFloat(always_apply=True),
            ToTensorV2(),
        ], additional_targets={'image2': 'image'})

        self.val_transforms = A.Compose([
            A.Resize(height, width, always_apply=True),
            A.ToFloat(always_apply=True),
            ToTensorV2(),
        ], additional_targets={'image2': 'image'})

    def __len__(self) -> int:
        return len(self.video_names)

    def __getitem__(self, index: int) -> torch.Tensor:
        video_name = self.video_names[index]

        # read video frames from random position
        frames = []
        idx1 = random.randint(0, self.video_dict[video_name] - self.max_frame_diff - 1)
        idx2 = idx1 + random.randint(1, self.max_frame_diff)

        video_path = os.path.join(self.data_root, f'JPEGImages/{video_name}.zip')
        img1 = ZipReader.imread(video_path, idx1).convert('RGB')
        img2 = ZipReader.imread(video_path, idx2).convert('RGB')
        img1 = np.array(img1)
        img2 = np.array(img2)

        # preprocess
        if self.is_train:
            transformed = self.train_transforms(image=img1, image2=img2)
            img1_t = transformed['image']
            img2_t = transformed['image2']
        else:
            transformed = self.val_transforms(image=img1, image2=img2)
            img1_t = transformed['image']
            img2_t = transformed['image2']

        return img1_t, img2_t