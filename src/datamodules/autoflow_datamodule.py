from typing import Any, Dict, Optional

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Sintel

from src.datamodules.components.autoflow_dataset import AutoFlowDataset


def sintel_transforms(img1, img2, flow, valid_flow_mask=None):
    """DO NOT RESIZE FROM 436x1024."""
    totensor = T.ToTensor()
    return totensor(img1), totensor(img2), torch.from_numpy(flow), None


class AutoFlowDataModule(LightningDataModule):
    """A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        train_data_dir: str,
        val_data_dir: str,
        test_data_dir: str,
        layer_num: int = 3,
        width: int = 512,
        height: int = 512,
        gaussian_blur_prob: float = 0.2,
        motion_blur_prob: float = 0.2,
        fog_prob: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if stage == "fit" and not self.data_train and not self.data_val:
            self.data_train = AutoFlowDataset(
                self.hparams.train_data_dir,
                self.hparams.layer_num,
                self.hparams.width,
                self.hparams.height,
                self.hparams.gaussian_blur_prob,
                self.hparams.motion_blur_prob,
                self.hparams.fog_prob,
            )
            self.data_val = Sintel(
                self.hparams.val_data_dir,
                split="train",
                pass_name="clean",
                transforms=sintel_transforms,
            )
        if stage == "test" and not self.data_test:
            self.data_test = Sintel(
                self.hparams.test_data_dir,
                split="train",  # test is not accompanied by GT flow
                pass_name="clean",
                transforms=sintel_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "autoflow.yaml")
    cfg.train_data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
