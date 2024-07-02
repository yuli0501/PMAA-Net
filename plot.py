from __future__ import annotations
import os

import cv2
import numpy as np
import torch

from typing import Sequence, Dict

# def is_grayscale(image: np.ndarray | torch.Tensor) -> bool:
#     return not (len(image.shape) > 2 and image.shape[2] > 1)

def save_x_y(
    x: np.ndarray, y: np.ndarray, out: str, color: Sequence[int] = (0, 0, 255)
) -> None:
    """
    input ndarray shape:
        x: [h, w, [c]]; y: [h, w];
    """
    assert all([x.dtype == np.uint8, y.dtype == np.uint8])
    #x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if is_grayscale(x) else cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    # y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
   # x = np.where(y > 0, np.full_like(x, color), x)
    cv2.imwrite(out, x)

# def save_x_y_tensor(x: torch.Tensor, y: torch.Tensor, out: str, **kwargs) -> None:
#     """
#     input ndarray shape:
#         x: [h, w, [c]]; y: [h, w];
#     """
#     x = x if is_grayscale(x) else x.permute(1, 2, 0)
#     x = x.detach().cpu().numpy().astype(np.uint8)
#     y = y.detach().cpu().numpy().astype(np.uint8)
#     save_x_y(x, y, out, **kwargs)

def save_batch_x_y(batch: Dict[str, ...], pred: torch.Tensor, out: str, scores):
    from utils import denormalize
    image = batch["image"][:, :, :, :].detach().cpu().numpy()
    image = [denormalize(e) for e in image]
    label = batch["label"][:, 0, :, :].detach().cpu().numpy().astype(np.uint8)
    case_names = batch["case_name"]
    print("##################")
    pred = pred[:, 0, :, :].detach().cpu().numpy().astype(np.uint8)
    for i in range(pred.shape[0]):
        save_x_y(
            x=image[i], y=pred[i],
            out=os.path.join(out, f"{case_names[i]}_pred_{scores[i]:.2f}.png"),
        )
        # save_x_y(
        #     x=image[i], y=label[i],
        #     out=os.path.join(out, f"{case_names[i]}_label.png"),
        # )
