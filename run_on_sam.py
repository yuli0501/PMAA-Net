from __future__ import annotations
import os
from os.path import join, splitext, basename
import math

import cv2
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch
import numpy as np
from monai.data import decollate_batch
from monai import transforms as T
from monai import metrics
from _dataset import load_isic_test_dataset
from models import UNet, SegFormer
from utils import cv2_rgb_loader, cv2_gray_loader, load_image, plot_sam_prediction

def compute_bbox_from_mask(mask: np.ndarray, threshold=5) -> tuple[int, int, int, int]:
    """
    >>> from utils import cv2_gray_loader, plot_mask_bbox
    >>> filename = '../../resource/datasets/ISIC-2017/256x256/train/label/18.png'
    >>> label = cv2_gray_loader(filename)
    >>> bbox = compute_bbox_from_mask(label)
    >>> plot_mask_bbox(label, bbox)
    """
    H, W = mask.shape[:2]
    x1, y1, x2, y2 = 0, 0, W, H
    for i in range(H):
        if np.sum(mask[i, :] > 0) > threshold:
            y1 = i
            break
    for i in range(W):
        if np.sum(mask[:, i] > 0) > threshold:
            x1 = i
            break
    for i in range(1, H + 1):
        if np.sum(mask[-i, :] > 0) > threshold:
            y2 = H - i
            break
    for i in range(1, W + 1):
        if np.sum(mask[:, -i] > 0) > threshold:
            x2 = W - i
            break
    return x1, y1, x2, y2

def compute_point_from_mask(mask: np.ndarray,
                            kernel_size: int | tuple[int, int] = (16, 16),
                            sample_times: int | tuple[float, float] = (0.5, 0.5),
                            threshold: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    # noinspection PyTypeChecker
    """
    >>> from utils import cv2_gray_loader, plot_mask_points
    >>> filename = '../../resource/datasets/ISIC-2017/256x256/train/label/18.png'
    >>> label = cv2_gray_loader(filename)
    >>> points, labels = compute_point_from_mask(label)
    >>> plot_mask_points(label, points)
    """
    def compute_pad_size(s, k):
        if s % k != 0:
            p = math.ceil(s / k) * k - s
            l = int(p / 2)
            return l, p - l
        return 0, 0

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    threshold = (kernel_size[0] * kernel_size[1]) * threshold
    if isinstance(sample_times, float):
        sample_times = (sample_times, sample_times)

    assert np.unique(mask).sum() == 255
    mask[mask == 255] = 1
    H, W = mask.shape[:2]
    # up, lp = compute_pad_size(H, kernel_size[0])
    # lp, rp = compute_pad_size(W, kernel_size[1])
    # mask = np.pad(mask[:, :, 0], ((up, lp), (lp, rp)), constant_values=0)
    # H, W = mask.shape[:2]

    s1 = H // int((H // kernel_size[0]) * sample_times[0])
    s2 = W // int((W // kernel_size[1]) * sample_times[1])

    points, labels = [], []
    for i in range(0, H, s1):
        for j in range(0, W, s2):
            patch = mask[j: j + kernel_size[1], i: i + kernel_size[0]]
            points.append([i + kernel_size[0] // 2, j + kernel_size[0] // 2])
            if np.sum(patch) >= threshold:
                labels.append(1)
            else:
                labels.append(0)
    # noinspection PyTypeChecker
    return np.array(points), np.array(labels)

def compute_random_points(mask: np.ndarray, num_samples=100, kernel_size=16) -> tuple[np.ndarray, np.ndarray]:
    H, W = mask.shape[:2]
    c = 0
    points, labels = [], []
    threshold = (kernel_size * kernel_size) * 0.75
    while c < num_samples:
        x = np.random.randint(low=0, high=H)
        y = np.random.randint(low=0, high=W)
        patch = mask[x: x + kernel_size, y: y + kernel_size]
        if np.sum(patch > 0) >= threshold:
            points.append([x, y])
            labels.append(1)
        else:
            # points.append([x, y])
            # labels.append(0)
            pass
        c += 1
    return np.array(points), np.array(labels)

def load_sam_predictor(
        checkpoint='models/segment_anything/pretrain/sam_vit_b_01ec64.pth',
        model_type='vit_b', device='cuda:0'):
    """
    >>> assert load_sam_predictor(device='cpu')
    """
    from models.segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def load_med_sam_predictor(
        checkpoint='models/medical_sam_adapter/pretrain/sam_vit_b_01ec64.pth',
        model_type='vit_b', device='cuda:0'):
    """
    >>> assert load_med_sam_predictor(device='cpu')
    """
    from easydict import EasyDict
    from models.medical_sam_adapter import sam_model_registry, SamPredictor

    args = EasyDict({
        'image_size': 256,
        'multimask_output': 1,
        'mod': 'sam_adpt',
        'mid_dim': None,
        'thd': False,
        'chunk': 96,
    })

    sam = sam_model_registry[model_type](args=args, checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def load_unet_model(checkpoint='log/unet/checkpoints/epoch=104,dice=0.92.pth', device='cuda:0'):
    """
    >>> assert load_unet_model(device='cpu')
    """
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model

def load_segformer_model(checkpoint='log/segformer/checkpoints/epoch=7,dice=0.90.pth', model_type='b1', device='cuda:0'):
    """
    >>> assert load_segformer_model(device='cpu')
    """
    model = SegFormer(phi=model_type).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model

def generate_pseudo_label(model, test_dataloader, device='cuda:0', output='log/pseudo_label'):
    os.makedirs(output, exist_ok=True)
    post_pred = T.Compose([T.Activations(sigmoid=True), T.AsDiscrete(threshold=0.5)])
    for image, filenames in tqdm(test_dataloader):
        image = image.to(device)
        with torch.no_grad():
            pred = post_pred(model(image))
            for mask, filename in zip(decollate_batch(pred), decollate_batch(filenames)):
                mask = mask.squeeze(0).detach().cpu().numpy()
                mask *= 255
                mask = mask.astype(np.uint8)
                Image.fromarray(mask).save(join(output, splitext(filename)[0] + '.png'))

def run_on_sam(predictor, dataset_folder, pseudo_label_folder, output='log/sam'):
    dice_metric = metrics.DiceMetric()
    jaccard_metric = metrics.MeanIoU()
    sp_metric = metrics.ConfusionMatrixMetric(metric_name=('sensitivity', 'specificity'))
    hausdorff_metric = metrics.HausdorffDistanceMetric()

    def pre_label(mask: np.ndarray) -> torch.Tensor:
        assert np.sum(np.unique(mask)) == 255, \
            'normalize that a mask pixel only be one of 0 or 255'
        mask[mask == 255] = 1
        mask = mask.astype(np.float32)
        mask = mask.transpose(2, 0, 1)
        return torch.from_numpy(mask).unsqueeze(0)

    def post_pred(pred: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)

    os.makedirs(output, exist_ok=True)
    for filename in tqdm(glob(join(dataset_folder, 'image', '*.*'))):
        image = load_image(filename, cv2_rgb_loader)
        filename = splitext(basename(filename))[0]
        label = load_image(join(dataset_folder, 'label', f'{filename}.*'), cv2_gray_loader)
        pseudo_label = load_image(join(pseudo_label_folder, f'{filename}.png'), cv2_gray_loader)

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)[:, :, None]
        pseudo_label = cv2.resize(pseudo_label, (256, 256), interpolation=cv2.INTER_NEAREST)[:, :, None]

        # bbox = compute_bbox_from_mask(pseudo_label)
        # points, labels = compute_point_from_mask(pseudo_label)
        # points, labels = compute_random_points(pseudo_label)
        predictor.set_image(image)
        pred, _, _ = predictor.predict(
            # box=np.array(bbox),
            # point_coords=np.array(points),
            # point_labels=labels,
            multimask_output=False)

        pred_tensor, label_tensor = post_pred(pred), pre_label(label)
        dice_metric(y_pred=pred_tensor, y=label_tensor)
        jaccard_metric(y_pred=pred_tensor, y=label_tensor)
        sp_metric(y_pred=pred_tensor, y=label_tensor)
        hausdorff_metric(y_pred=pred_tensor, y=label_tensor)

        pred = pred.astype(np.float32).transpose((1, 2, 0)) * 255
        plot_sam_prediction(
            image=image,
            label=label,
            pseudo_label=pseudo_label,
            pred=pred,
            # bbox=bbox,
            # points=points,
            # labels=labels,
            output=join(output, f'{filename}.png')
        )

    mean_jaccard = jaccard_metric.aggregate().item()
    mean_sp = sp_metric.aggregate()
    mean_sensitivity = mean_sp[0].item()
    mean_specificity = mean_sp[1].item()
    mean_hausdorff = hausdorff_metric.aggregate().item()

    print(f'''
        mean_jaccard:     {mean_jaccard:.2f}
        mean_sensitivity: {mean_sensitivity:.2f}
        mean_specificity: {mean_specificity:.2f}
        mean_hausdorff:   {mean_hausdorff:.2f}
    ''')

if __name__ == '__main__':
    # Generate pseudo label
    # root = r'E:\ccw\_dataset\PH2_512x512\image'
    # generate_pseudo_label(
    #     models=load_segformer_model(),
    #     test_dataloader=load_isic_test_dataset(root).test_dataloader,
    #     output='log/pseudo_label_segformer'
    # )

    # bbox
    # mean_jaccard:     0.78
    # mean_sensitivity: 0.89
    # mean_specificity: 0.92
    # mean_hausdorff:   77.79

    # points
    # mean_jaccard:     0.63
    # mean_sensitivity: 0.93
    # mean_specificity: 0.71
    # mean_hausdorff:   164.54

    run_on_sam(
        predictor=load_sam_predictor(),
        # predictor=load_med_sam_predictor(),
        # predictor=load_sam_predictor('models/segment_anything/pretrain/sam_vit_l_0b3195.pth', model_type='vit_l'),
        dataset_folder=r'E:\ccw\dataset\PH2_512x512',
        pseudo_label_folder='log/pseudo_label_unet',
        output='log/sam/sam_zero_shot'
    )

    # score_arr = []
# os.makedirs('results', exist_ok=True)
# for img_path in tqdm(glob("E:/ccw/_dataset/ISIC2017_512x512/images/*.jpg")):
#     image = cv2.imread(img_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     ground_truth_path = os.path.join("E:/ccw/_dataset/ISIC2017_512x512/masks", f'{os.path.basename(img_path).split(".")[0]}.*')
#     ground_truth_path = glob(ground_truth_path)[0]
#     ground_truth = cv2.imread(ground_truth_path)
#     ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
#     boxes = box_points(ground_truth)
#
#     predictor.set_image(image)
#     mask, score, logit = predictor.predict(box=boxes, multimask_output=False)
#     score_arr.append(score)
#
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_box(boxes, plt.gca())
#     plt.title('Score: {}'.format(float(score)))
#     plt.axis('off')
#     plt.savefig(os.path.join('results', os.path.basename(img_path)))
#
# print(np.mean(score_arr))
