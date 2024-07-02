from __future__ import annotations
import random
import warnings
from albumentations import BasicTransform, DualTransform
import numpy as np
import cv2
from typing import Callable, Any
from utils import load_image, cv2_rgb_loader, cv2_gray_loader

class MaskOnlyTransform(BasicTransform):
    """Transform applied to mask only."""

    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        params = {k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()}
        return self.apply(mask, *args, **params)

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {'mask': self.apply_to_mask}

class MaskNormalize(MaskOnlyTransform):
    """
    >>> from utils import cv2_rgb_loader, cv2_gray_loader
    >>> filename = '../../../resource/datasets/ISIC-2017/256x256/train/image/18.jpg'
    >>> image = cv2_rgb_loader(filename)
    >>> filename = '../../../resource/datasets/ISIC-2017/256x256/train/label/18.png'
    >>> label = cv2_gray_loader(filename)
    >>> aug = MaskNormalize()
    >>> augmented = aug(image=image, mask=label)
    >>> assert np.all(image == augmented['image'])
    >>> np.unique(augmented['mask'])
    array([0, 1], dtype=uint8)
    """
    def __init__(self, dtype: np.dtype = np.float32, always_apply: bool = False, p: float = 1.0):
        super(MaskNormalize, self).__init__(always_apply, p)
        self.dtype = dtype

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        assert np.sum(np.unique(img)) == 255, \
            'normalize that a mask pixel only be one of 0 or 255'
        img[img == 255] = 1
        return img.astype(self.dtype)

def get_shuffled_args(h: int, w: int, patch_h: int = 16, patch_w: int = 16) \
        -> tuple[int, int, list, list]:
    """
    >>> from utils import plot_mask_points
    >>> _, _, o_coords, t_coords = get_shuffled_args(224, 320, 32, 32)
    >>> plot_mask_points(np.ones((224, 320)) * 255, t_coords)
    """
    if h % patch_h != 0 or w % patch_w != 0:
        warnings.warn('a given patch size cannot split the image into integer blocks')
    num_h, num_w = h // patch_h, w // patch_w
    a, b = list(range(num_h * num_w)), list(range(num_h * num_w))
    random.shuffle(b)
    origin_coords, target_coords = [], []
    for o, t in zip(a, b):
        origin_x, origin_y = (o // num_h) * patch_w, (o % num_h) * patch_h
        target_x, target_y = (t // num_h) * patch_w, (t % num_h) * patch_h
        origin_coords.append([origin_x, origin_y])
        target_coords.append([target_x, target_y])
    return patch_h, patch_w, origin_coords, target_coords

def shuffle_image_patch(img: np.ndarray, shuffled_args: tuple[int, int, list, list]) -> np.ndarray:
    """
    >>> from utils import cv2_rgb_loader, cv2_gray_loader, plot_image
    >>> filename = '../../../resource/datasets/ISIC-2018/224x320/image/ISIC_0000025.jpg'
    >>> image = cv2_rgb_loader(filename)
    >>> patch_size = (32, 32)
    >>> shuffled_args = get_shuffled_args(*image.shape[:2], *patch_size)
    >>> shuffled_image = shuffle_image_patch(image, shuffled_args)
    >>> plot_image(shuffled_image)
    >>> filename = '../../../resource/datasets/ISIC-2018/224x320/label/ISIC_0000025.png'
    >>> label = cv2_gray_loader(filename)
    >>> shuffled_label = shuffle_image_patch(label, shuffled_args)
    >>> plot_image(shuffled_label)
    """
    p_h, p_w, origin_coords, target_coords = shuffled_args
    t_img = np.zeros_like(img)
    for (o_x, o_y), (t_x, t_y) in zip(origin_coords, target_coords):
        t_img[t_y: t_y + p_h, t_x: t_x + p_w] = img[o_y: o_y + p_h, o_x: o_x + p_w]
    return t_img

class ShufflePatch(DualTransform):
    def __init__(self, patch_size: tuple[int, int], always_apply: bool = False, p: float = 0.5):
        super(ShufflePatch, self).__init__(always_apply, p)
        self.patch_size = patch_size
        self.shuffled_args = None

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        if self.shuffled_args is None:
            self.shuffled_args = get_shuffled_args(*img.shape[:2], *self.patch_size)
        return shuffle_image_patch(img, self.shuffled_args)

class LoadImage(DualTransform):
    def __init__(self, image_loader=None, mask_loader=None, always_apply: bool = False, p: float = 1):
        super(LoadImage, self).__init__(always_apply, p)
        self.image_loader = image_loader or (lambda e: load_image(e, cv2_rgb_loader))
        self.mask_loader = mask_loader or (lambda e: load_image(e, cv2_gray_loader))

    def apply(self, img: str, *args: Any, **params: Any) -> np.ndarray:
        return self.image_loader(img)

    def apply_to_mask(self, mask: str, *args: Any, **params: Any) -> np.ndarray:
        return self.mask_loader(mask)

class Identity(DualTransform):
    def __init__(self, always_apply: bool = False, p: float = 1):
        super(Identity, self).__init__(always_apply, p)

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return img
