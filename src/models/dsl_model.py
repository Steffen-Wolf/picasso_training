from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from mipnet.models.unet import UNet2d
from torch.nn import MSELoss

class SparseLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = MSELoss()

    def forward(self, preds, targets):
        mask = targets != -100
        # regularization_loss = self.loss(preds[~mask], targets[~mask])
        regression_loss = self.loss(preds[mask], targets[mask])

        return regression_loss# + 0.0000001 * regularization_loss

class DSLModel(LightningModule):

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 7,
        lr: float = 0.001,
        in_size=(256, 256),
        weight_decay: float = 0.0005
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = UNet2d(in_channels,
                            out_channels,
                            pad_convs=True)

        # loss function
        self.criterion = SparseLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log("val/loss", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}
        # return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        self.log("test/loss", loss.detach().item(), on_step=False, on_epoch=True)

        return {"loss": loss}
        # return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
