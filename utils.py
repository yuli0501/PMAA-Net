from __future__ import annotations
import random
import os
import sys

import numpy as np
import torch
import cv2
import warnings
from matplotlib import pyplot as plt
from PIL import Image
import json
from monai.data import decollate_batch
from glob import glob
import re

from typing import Callable, Any, Sequence
#
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cv2_rgb_loader(pathname: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(pathname, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#
def cv2_gray_loader(pathname: str) -> np.ndarray:
    return cv2.imread(pathname, cv2.IMREAD_GRAYSCALE)[:, :, None]
#
def pil_rgb_loader(pathname: str) -> Image.Image:
    return Image.open(pathname).convert("RGB")

def pil_gray_loader(pathname: str) -> Image.Image:
    return Image.open(pathname).convert("L")
#
def load_image(
    pathname: str,
    loader: Callable[..., Image.Image | np.ndarray] = cv2_rgb_loader
) -> Image.Image | np.ndarray:
    if any([e in pathname for e in ["*", "?"]]):
        filelist = glob(pathname)
        if len(filelist) == 0:
            raise FileNotFoundError(pathname)
        if len(filelist) > 1:
            warnings.warn("multiple images found in {}".format(pathname))
        pathname = filelist[0]
    return loader(pathname)
#
def normalize_ext(ext: str) -> str:
    if ext.strip() == "" or ext.startswith("."):
        return ext
    return "." + ext
#
def dc(result: torch.Tensor, reference: torch.Tensor) -> float:
    result = torch.atleast_1d(result.type(torch.bool))
    reference = torch.atleast_1d(reference.type(torch.bool))
    intersection = torch.count_nonzero(result & reference)

    size_i1 = torch.count_nonzero(result)
    size_i2 = torch.count_nonzero(reference)

    try:
        dc = (2. * intersection / float(size_i1 + size_i2)).item()
    except ZeroDivisionError:
        dc = 0.0
    return dc
#
def calc_dice_gpu(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    input tensor shape:
        pred: [[d,] h, w]; gt: [[d,] h, w]
    output tensor shape: [0]
    """
    if pred.sum() > 0 and gt.sum() > 0:
        return dc(pred, gt)
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1
    return 0
#
def make_color_lighter(rgb_color: Sequence[int], percentage: int) -> tuple[int, int, int]:
    red, green, blue = rgb_color
    red = max(0, int(red - (red * percentage / 100)))
    blue = max(0, int(blue - (blue * percentage / 100)))
    green = min(255, int(green + (255 * percentage / 100)))
    return red, green, blue
#
def save_ilp(
    image: np.ndarray, label: np.ndarray, pred: np.ndarray,
    output: str,
    color: Sequence[int] = (0, 0, 255),
    percentage: int = 10,
    denormalize_img: bool = False,
):
    """
    input tensor shape:
        image: [h, w]; label: [h, w]; pred: [h, w]
    """
    if denormalize_img:
        image = denormalize(image)
    else:
        image = image * 255
        image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    # noinspection PyTypeChecker
    # image = np.where(pred > 0, np.full_like(image, make_color_lighter(color, percentage)), image)
    # noinspection PyUnresolvedReferences
    # print(np.unique(pred), pred.shape)
    contours, _ = cv2.findContours((pred > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 1)
    contours, _ = cv2.findContours((label > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, [255, 0, 0], 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if os.path.dirname(output) != "":
        os.makedirs(os.path.dirname(output), exist_ok=True)
    cv2.imwrite(output, image)
#
def save_batch_ilp(batch: dict[str, ...], pred: torch.Tensor, output: str, **kwargs: Any):
    """Save a batch of images, labels, predictions to files.

    image: [b, 1, h, w]; label: [b, 1, h, w]; pred:  [b, 1, h, w]
    """
    image = batch["image"][:, :, :, :].detach().cpu().numpy()
    label = batch["label"][:, 0, :, :].detach().cpu().numpy().astype(np.uint8)
    case_names = batch["case_name"]
    pred = pred[:, 0, :, :].detach().cpu().numpy().astype(np.uint8)
    for i in range(pred.shape[0]):
        save_ilp(
            image=image[i], label=label[i], pred=pred[i],
            output=os.path.join(output, f"{case_names[i]}.png"),
            **kwargs
        )
#
class PathBuilder(object):
    def __init__(self, *pathnames: str, mkdir: bool = True) -> None:
        self.mkdir = mkdir
        self.path = str(os.path.join(*(pathnames or (".",))))
        if self.mkdir:
            os.makedirs(self.path, exist_ok=True)

    def __truediv__(self, pathname: str) -> 'PathBuilder':
        return PathBuilder(os.path.join(self.path, pathname), mkdir=self.mkdir)

    def __add__(self, basename: str) -> str:
        return os.path.join(self.path, basename)

    def __repr__(self) -> str:
        return self.path
#
def plot_image_mask(image: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> None:
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        if len(mask.shape) == 3:
            mask = mask.permute(1, 2, 0)
        mask = mask.detach().cpu().numpy()

    fig = plt.figure()
    axes = fig.subplots(1, 2)
    for ax, img in zip(axes.flatten(), (image, mask)):
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.imshow(img)
    plt.show()

def save_batch_image_mask(image: torch.Tensor, mask: torch.Tensor, root: str):
    image = [denormalize(e) for e in decollate_batch(image)]
    label = [(e.cpu().numpy() * 255).astype(np.uint8)
             for e in decollate_batch(mask.permute(0, 2, 3, 1))]
    for b in range(len(image)):
        fig = plt.figure()
        axes = fig.subplots(1, 2)
        for ax, img in zip(axes.flatten(), (image[b], label[b])):
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.imshow(img)
        plt.savefig(os.path.join(root, f'batch_{b}.png'))
        plt.close(fig)

def plot_mask_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]):
    plt.figure()
    plt.imshow(mask)
    x0, y0 = bbox[0], bbox[1]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    plt.show()

def plot_mask_points(mask: np.ndarray, points: list[list[int, int]]):
    plt.figure()
    plt.imshow(mask)
    ax = plt.gca()
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], color='red', marker='*', s=20)
    plt.show()

def plot_sam_prediction(image, label, pseudo_label, pred, output, bbox=None, points=None, labels=None):
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    # 绿
    label *= np.array([[0, 255, 0]], dtype=np.uint8)
    pseudo_label = cv2.cvtColor(pseudo_label, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    # 蓝
    pseudo_label *= np.array([[0, 0, 1]], dtype=np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    # 红
    pred *= np.array([[1, 0, 0]], dtype=np.uint8)
    # g = cv2.addWeighted(image, 1, pseudo_label, 1, 0)
    g = cv2.addWeighted(image, 1, label, 0.5, 0)
    g = cv2.addWeighted(g, 1, pred, 0.8, 0)
    plt.figure()
    plt.imshow(g)
    ax = plt.gca()
    if bbox is not None:
        x0, y0 = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    if points is not None and labels is not None:
        points = np.array(points)
        ax.scatter(points[:, 0][labels == 0], points[:, 1][labels == 0], color='red', marker='*', s=20)
        ax.scatter(points[:, 0][labels == 1], points[:, 1][labels == 1], color='green', marker='o', s=20)
    plt.savefig(output)
    plt.close()

def denormalize(
    image: torch.Tensor | np.ndarray,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_value: float = 255.0
) -> np.ndarray:
    """
    input tensor shape: [c, h, w]
    output ndarray shape: [h, w, c]
    """
    assert len(image.shape) == 3
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    if image.shape[0] <= 3:
        image = np.transpose(image, (1, 2, 0))
    mean = np.array(mean, dtype=np.float32) * max_pixel_value
    std = np.array(std, dtype=np.float32) * max_pixel_value
    return (image * std + mean).astype(np.uint8)

def random_click_point(mask, point_labels=1):
    """
    >>> mask = np.array([
    ... [0, 0, 1],
    ... [0, 1, 1],
    ... [1, 1, 1]])
    >>> random_click_point(mask)
    """
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label)
    print(indices)
    return point_labels, indices[np.random.randint(len(indices))]

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def filename(path: str) -> str:
    """
    >>> filename('./ISIC2018_Task1-2_Training_Input/ISIC_0000025.jpg')
    'ISIC_0000025'
    """
    return os.path.basename(os.path.splitext(path)[0])
#
def load_json(pathname: str, encoding: str | None = None) -> Any:
    with open(pathname, "r", encoding=encoding or sys.getdefaultencoding()) as fp:
        return json.load(fp)
#
def plot_image(img: np.ndarray | torch.Tensor, figsize=None, dpi=None, cmap=None):
    if isinstance(img, torch.Tensor):
        if len(img.shape) > 2:
            img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img, cmap=cmap)
    plt.show()

def find_best_checkpoint(root: str, metric_name='dice'):
    path = glob(os.path.join(root, '*.pth'))
    index = np.argsort([
        float(re.findall(f'.*?{metric_name}=([0-9]+\\.[0-9]+).*', e)[0])
        for e in path
    ])[-1]
    return path[index]

def get_flops_params(model, input_size=(1, 3, 224, 224)):
    from thop import profile

    input = torch.randn(input_size)
    flops, params = profile(model, (input,))
    print('FLOPs: ' + str(flops / 1000 ** 3) + 'G')
    print('Params: ' + str(params / 1000 ** 2) + 'M')
