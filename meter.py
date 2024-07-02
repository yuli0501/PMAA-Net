from __future__ import annotations

import numpy as np
import torch
from medpy.metric import binary as metric

from utils import calc_dice_gpu

np.bool = bool

class BinarySegMeter(object):
    def __init__(self, threshold: float = 0.5) -> None:
        self.reset()
        self.threshold = threshold

    def reset(self) -> None:
        self.metric = {
            "dice": ([], calc_dice_gpu),
        }

    def __call__(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """
        input tensor shape:
            input: [b, 1, h, w]; target: [b, 1, h, w]
        """
        for batch_idx in range(pred.shape[0]):
            y_hat, y = pred[batch_idx], label[batch_idx]
            for _, (arr, metric) in self.metric.items():
                arr.append(metric(
                    torch.asarray(y_hat > self.threshold, dtype=torch.int),
                    torch.asarray(y, dtype=torch.int)
                ))

    def get_metric(self) -> dict[str, list[float]]:
        """
        output tensor shape:
            {
                "metric name": [val1, val2, ...], ...
            }
        """
        result = {}
        for metric_name, (v, _) in self.metric.items():
            result[metric_name] = v
        return result

class BinarySegMeterCPU(object):
    def __init__(self, threshold: float = 0.5) -> None:
        self.reset()
        self.threshold = threshold

    def reset(self) -> None:
        self.metric = {
            "dice": ([], metric.dc),
            "iou": ([], metric.jc),
            "hd95": ([], metric.hd95),
            "specificity": ([], metric.specificity),
            "sensitivity": ([], metric.sensitivity),
        }

    def __call__(self, pred: torch.Tensor | np.ndarray, label: torch.Tensor | np.ndarray):
        """
        input tensor/ndarray shape:
            input: [b, 1, h, w]; target: [b, 1, h, w]
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()

        for batch_idx in range(pred.shape[0]):
            y_hat, y = pred[batch_idx], label[batch_idx]
            for name, (arr, metric) in self.metric.items():
                arr.append(metric(
                    np.asarray(y_hat > self.threshold, dtype=np.int32),
                    np.asarray(y, dtype=np.int32)
                ))

    def get_metric(self) -> dict[str, list[float]]:
        """
        output data formation:
            { "metric name": [val1, val2, ...], ... }
        """
        result = {}
        for metric_name, (v, _) in self.metric.items():
            result[metric_name] = v
        return result
