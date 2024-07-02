from __future__ import annotations

from os.path import join, basename, splitext
from glob import glob
from functools import partial

import numpy as np
from torch.utils.data import Dataset

from loguru import logger
from typing import Callable, Sequence, Any, Literal

from transforms import A
from utils import load_image, cv2_gray_loader, load_json, normalize_ext

class BaseDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sequence[str, str]],
        loader: Callable[..., np.ndarray] | Sequence[Callable[..., np.ndarray]],
        transform: A.TransformType | None = None,
    ):
        super(BaseDataset, self).__init__()
        self.samples = tuple(samples)
        self.loaders = (loader[0], loader[1]) if isinstance(loader, Sequence) else (loader, loader)
        self.transform = transform
        logger.info(f"Loading dataset with {len(self.samples)} examples")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, label = self.samples[idx]
        case_name = splitext(basename(image))[0]
        image, label = self.loaders[0](image), self.loaders[1](label)
        assert image.shape[:2] == label.shape[:2], \
            "the spacial resolution of a image and its label does not match"
        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]
        return {"case_name": case_name, "image": image, "label": label}

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> dict[str, Any]:
        for i in range(len(self)):
            yield self[i]

class ISICDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform: A.TransformType | None = None,
        image_folder_name: str = "image", label_folder_name: str = "label",
        image_ext: str = ".*", label_ext: str = ".*",
    ):
        image_root, label_root = join(root, image_folder_name), join(root, label_folder_name)
        images = glob(join(image_root, f"*{normalize_ext(image_ext)}"))
        labels = [join(label_root, f"{basename(splitext(e)[0])}{normalize_ext(label_ext)}") for e in images]
        super(ISICDataset, self).__init__(
            samples=zip(images, labels),
            loader=(load_image, partial(load_image, loader=cv2_gray_loader)),
            transform=transform
        )

class ISICKFoldDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        kfold_file_name: str, k: int = 1,
        mode: Literal["training", "validation", "testing"] = "training",
        transform: A.TransformType = None,
        image_folder_name: str = "image", label_folder_name: str = "label",
        image_ext: str = ".*", label_ext: str = ".*",
    ):
        image_root, label_root = join(root, image_folder_name), join(root, label_folder_name)
        fold = load_json(join(root, kfold_file_name))[f"folder{k}"]
        samples = [(join(image_root, f"{e}{image_ext}"), join(label_root, f"{e}{label_ext}")) for e in fold[mode]]
        super(ISICKFoldDataset, self).__init__(
            samples=samples,
            loader=(load_image, partial(load_image, loader=cv2_gray_loader)),
            transform=transform
        )
