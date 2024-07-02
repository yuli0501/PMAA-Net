from __future__ import annotations

import os
from os.path import join

from lightning.pytorch.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS
import torch
import numpy as np

import monai
from monai import data
from monai.metrics import CumulativeAverage
from thop import profile
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from loguru import logger
from typing import Callable

import lr_scheduler
from loss import DiceBCEWithLogitsLoss

from typing import Any

from meter import BinarySegMeter, BinarySegMeterCPU
from utils import save_batch_ilp, PathBuilder

# ====================================================================================== #

from models.segformer import build_segformer_b0 as build_model
exp_name: str = "segformer_b0_v1_module_2_talking_heads"

# from models.emh import build_emhformer as build_model
# exp_name: str = "emhformer"

# ====================================================================================== #

torch.set_float32_matmul_precision("medium")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "AdamW": torch.optim.AdamW
}

LR_SCHEDULERS = {
    "PolynomialLR": lr_scheduler.PolynomialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts
}

LOSSES = {
    "DiceCELoss": DiceBCEWithLogitsLoss,
}

class ISIC(L.LightningModule):
    def __init__(self, name: str = "test") -> None:
        super(ISIC, self).__init__()
        self.name: str = name

        self.max_epochs: int = 100
        logger.info(f"Max epochs: {self.max_epochs}")

        self._model = build_model().to(device)
        logger.info(f"Using models: {type(self._model).__name__}")

        self.build_loss()

        self.tl_metric = CumulativeAverage()
        self.vs_metric = BinarySegMeter(threshold=0.5)
        self.ts_metric = BinarySegMeterCPU(threshold=0.5)
        self.test_img_root = PathBuilder(self.name, "test_img")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def prepare_data(self) -> None:
        from transforms import A, ToTensorV2, Mask2Label
        from dataset import ISICKFoldDataset

        train_transform0 = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-40, 40)),
            A.Normalize(),
            Mask2Label(),
            ToTensorV2(transpose_mask=True)
        ])

        val_transform0 = A.Compose([
            A.Normalize(),
            Mask2Label(),
            ToTensorV2(transpose_mask=True)
        ])

        root = r"E:\ccw\dataset\ISIC2018_224x320"
        logger.info(f"Loading data root: {root}")

        kfold_file_name = os.path.join(root, "ISIC-2018-5-folds.json")
        logger.info(f"Loading kfold file: {kfold_file_name}")

        k = 1
        logger.info(f"Loading data {k}-fold data")

        train_transform = train_transform0
        logger.info(f"Training transform: {train_transform}")
        self.train_dataset = ISICKFoldDataset(
            root=root,
            kfold_file_name=kfold_file_name, k=k, mode="training",
            transform=train_transform,
        )

        val_transform = val_transform0
        logger.info(f"Validation transform: {val_transform}")
        self.val_dataset = ISICKFoldDataset(
            root=root,
            kfold_file_name=kfold_file_name, k=k, mode="validation",
            transform=val_transform,
        )

        test_transform = val_transform0
        logger.info(f"Testing transform: {val_transform}")
        self.test_dataset = ISICKFoldDataset(
            root=root,
            kfold_file_name=kfold_file_name, k=k, mode="testing",
            transform=test_transform,
        )

    def train_dataloader(self) -> data.DataLoader:
        tdl_0 = {
            "batch_size": 32,
            "num_workers": 8,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
        }

        tdl = tdl_0
        logger.info(f"Training dataloader: {tdl}")
        return data.DataLoader(self.train_dataset, **tdl)

    def val_dataloader(self) -> data.DataLoader:
        vdl_0 = {
            "batch_size": 16,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 8,
            "persistent_workers": True
        }

        vdl = vdl_0
        logger.info(f"Validation dataloader: {vdl}")
        return data.DataLoader(self.val_dataset, **vdl)

    def test_dataloader(self) -> data.DataLoader:
        tdl_0 = {
            "batch_size": 16,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 8,
            "persistent_workers": True
        }

        tdl = tdl_0
        logger.info(f"Testing dataloader: {tdl}")
        return data.DataLoader(self.test_dataset, **tdl)

    def build_loss(self):
        loss_0 = ("DiceCELoss", {
            "ce_weight": 0.4,
            "dc_weight": 0.6,
        })

        loss_1 = ("DiceCELoss", {
            "ce_weight": 1.0,
            "dc_weight": 1.0,
        })

        loss_config = loss_1
        logger.info(f"Using loss: {loss_config}")
        self.loss = LOSSES[loss_config[0]](**loss_config[1])

    @property
    def criterion(self) -> Callable[..., torch.Tensor]:
        return self.loss

    def configure_optimizers(self) -> dict:
        optimizer_0 = ("AdamW", {
            "lr": 0.00085,
            "weight_decay": 1e-4,
        })

        optimizer_config = optimizer_0
        optimizer = OPTIMIZERS[optimizer_config[0]](self._model.parameters(), **optimizer_config[1])
        logger.info(f"Using optimizer: {optimizer_config}")

        scheduler_0 = ("PolynomialLR", {
            "total_iters": self.max_epochs,
            "power": 0.9
        })

        scheduler_config = scheduler_0
        scheduler = LR_SCHEDULERS[scheduler_config[0]](optimizer, **scheduler_config[1])
        logger.info(f"Using lr scheduler: {scheduler_config}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def log_and_logger(self, name: str, value: Any) -> None:
        self.log(name, value)
        logger.info(f"{name}: {value}")

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        image, label = batch["image"].to(device), batch["label"].to(device)

        pred = self.forward(image)
        loss = self.criterion(pred, label)

        self.log("loss", loss.item(), prog_bar=True)
        self.tl_metric.append(loss.item())
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log_and_logger("mean_train_loss", self.tl_metric.aggregate().item())
        self.tl_metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor]) -> None:
        image, label = batch["image"].to(device), batch["label"].to(device)

        pred = torch.sigmoid(self.forward(image))  # [b, 1, h, w]
        self.vs_metric(pred, label)

    def on_validation_epoch_end(self) -> None:
        for metric_name, value in self.vs_metric.get_metric().items():
            self.log_and_logger(f"val_mean_{metric_name}", np.mean(value))
        self.vs_metric.reset()

    def test_step(self, batch: dict[str, torch.Tensor]) -> None:
        image, label = batch["image"].to(device), batch["label"].to(device)

        pred = torch.sigmoid(self.forward(image))  # [b, 1, h, w]
        self.ts_metric(pred, label)
        save_batch_ilp(
            batch=batch, pred=pred,
            output=self.test_img_root.path
        )

    def on_test_epoch_end(self) -> None:
        for metric_name, value in self.ts_metric.get_metric().items():
            self.log_and_logger(f"test_mean_{metric_name}", np.mean(value))
        self.vs_metric.reset()

def train(name: str) -> None:
    os.makedirs(name, exist_ok=True)
    logger.add(join(name, "training.log"))

    logger.info(f"Exp name: {exp_name}")
    model = ISIC(name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=join(name, "checkpoints"),
        monitor="val_mean_dice",
        mode="max",
        filename="{epoch:02d}-{val_mean_dice:.2f}",
        save_last=True
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="mean_train_loss",
    #     mode="min",
    #     min_delta=0.00,
    #     patience=15
    # )

    trainer = L.Trainer(
        precision=32,
        accelerator=device,
        devices="auto",
        max_epochs=model.max_epochs,
        check_val_every_n_epoch=20,
        gradient_clip_val=None,
        default_root_dir=name,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
    )

    trainer.fit(model, ckpt_path=None)

def train_full(name: str, seed: int = 42) -> None:
    L.seed_everything(seed)
    monai.utils.set_determinism(seed)
    train(name)

if __name__ == "__main__":
    train_full(f"log/{exp_name}")
    model = ISIC(f"log/{exp_name}")
    # model = ISIC.load_from_checkpoint(f"./log/{exp_name}/checkpoints/epoch=99-val_mean_dice=0.89.ckpt")
    # trainer = L.Trainer()
    # trainer.test(model)
    input_size = (1, 3, 224, 224)
    input = torch.randn(input_size)
    flops, params = profile(model, (input,))
    print('FLOPs: ' + str(flops / 1000 ** 3) + 'G')
    print('Params: ' + str(params / 1000 ** 2) + 'M')
