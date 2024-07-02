from __future__ import annotations

import numpy as np
import cv2

# noinspection PyUnresolvedReferences
import albumentations as A
# noinspection PyUnresolvedReferences
from albumentations.pytorch import ToTensorV2
from albumentations import BasicTransform

from typing import Callable, Any, Tuple

class MaskOnlyTransform(BasicTransform):
    """Transform applied to mask only."""

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        params = {k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()}
        return self.apply(mask, *args, **params)

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {"mask": self.apply_to_mask}

class Mask2Label(MaskOnlyTransform):
    def __init__(self, dtype: np.dtype = np.float32, always_apply: bool = False, p: float = 1.0):
        super(Mask2Label, self).__init__(always_apply, p)
        self.dtype = dtype

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        assert np.sum(np.unique(img)) == 255, \
            "normalize that a mask pixel only be one of 0 or 255"
        img[img == 255] = 1
        return img.astype(self.dtype)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("dtype",)
