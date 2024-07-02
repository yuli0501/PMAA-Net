from __future__ import annotations

import torch
from torch.optim.lr_scheduler import *

class LinearWarmupPolyLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_ratio: float,
        max_epochs: int,
        power: float = 1.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio
        self.total_iters = max_epochs - warmup_epochs
        self.power = power
        super(LinearWarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] * (1. + self.warmup_ratio) for group in self.optimizer.param_groups]
        else:
            decay_factor = ((1.0 - self.last_epoch / self.total_iters) /
                            (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
            return [group['lr'] * decay_factor for group in self.optimizer.param_groups]
