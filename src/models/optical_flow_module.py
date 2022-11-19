from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import flow_to_image

from src.models.components.metric import EPE


class OpticalFlowModule(LightningModule):
    """A LightningModule organizes your PyTorch code into 6 sections:

        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        weights: list[float],
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.L1Loss()

        # metric function
        self.epe = EPE()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        return self.net(img1, img2)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        pass

    def step(self, batch: Any):
        img1, img2, gt_flow = batch
        pred_list = self.forward(img1, img2)

        # loss (be aware to align the cale of flow vectors)
        assert len(pred_list) <= len(self.hparams.weights)
        loss = 0
        for i, (pred, w) in enumerate(zip(pred_list, self.hparams.weights)):
            gt_small = F.interpolate(gt_flow, pred.shape[-2:], mode="bilinear")
            loss += self.criterion(pred * (i + 1), gt_small) * w

        pred_out = pred_list[0]
        pred_out[:, 0, :, :] *= gt_flow.shape[-1] / pred_out.shape[-1]
        pred_out[:, 1, :, :] *= gt_flow.shape[-2] / pred_out.shape[-2]
        pred_out = F.interpolate(pred_out, size=gt_flow.shape[-2:], mode="bilinear")

        return loss, pred_out

    def training_step(self, batch: Any, batch_idx: int):
        img1, img2, gt_flow = batch
        loss, pred_flow = self.step(batch)

        # log train metrics
        epe = self.epe(pred_flow, gt_flow)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/epe", epe, on_step=False, on_epoch=True, prog_bar=True)

        # log images
        if batch_idx == 0 and isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_images(
                "train/image1",
                torch.clip(img1 * 255, 0, 255).to(torch.uint8),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                "train/image2",
                torch.clip(img2 * 255, 0, 255).to(torch.uint8),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                "train/pred", flow_to_image(pred_flow), self.current_epoch
            )
            self.logger.experiment.add_images(
                "train/gt",
                flow_to_image(gt_flow),
                self.current_epoch,
            )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return loss

    def training_epoch_end(self, outputs: list[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        img1, img2, gt_flow = batch
        loss, pred_flow = self.step(batch)

        # log val metrics
        epe = self.epe(pred_flow, gt_flow)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/epe", epe, on_step=False, on_epoch=True, prog_bar=True)

        # log images
        if batch_idx == 0 and isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_images(
                "val/image1",
                torch.clip(img1 * 255, 0, 255).to(torch.uint8),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                "val/image2",
                torch.clip(img2 * 255, 0, 255).to(torch.uint8),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                "val/pred",
                flow_to_image(pred_flow),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                "val/gt",
                flow_to_image(gt_flow),
                self.current_epoch,
            )

        return loss

    def validation_epoch_end(self, outputs: list[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        img1, img2, gt_flow = batch
        loss, pred_flow = self.step(batch)

        # log test metrics
        epe = self.epe(pred_flow, gt_flow)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/epe", epe, on_step=False, on_epoch=True)

        return loss

    def test_epoch_end(self, outputs: list[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            optimizer.param_groups[0]["lr"],
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "upflownet_light.yaml")
    _ = hydra.utils.instantiate(cfg)
