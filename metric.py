import torch
from torch import Tensor
from sklearn.metrics import roc_curve, auc

def confusion_matrix(input: Tensor, target: Tensor, threshold: float = 0.5):
    input = (torch.sigmoid(input) > threshold).to(torch.float32)
    dim = list(range(len(input.shape))[1:])
    TP = torch.sum((input == 1) & (target == 1), dim=dim)
    FP = torch.sum((input == 1) & (target == 0), dim=dim)
    TN = torch.sum((input == 0) & (target == 0), dim=dim)
    FN = torch.sum((input == 0) & (target == 1), dim=dim)
    return TP, FP, TN, FN

def iou(cm, eps=1e-6):
    TP, FP, _, FN = cm
    return ((TP + eps) / (TP + FN + FP + eps)).mean()

def jaccard_index(cm, eps=1e-7):
    return iou(cm, eps)

def dice_coef(cm, eps=1e-7):
    TP, FP, _, FN = cm
    return ((2 * TP + eps) / (2 * TP + FN + FP + eps)).mean()

def accuracy(cm, eps=1e-7):
    TP, FP, TN, FN = cm
    return ((TP + TN + eps) / (TP + FP + TN + FN + eps)).mean()

def precision(cm, eps=1e-7):
    TP, FP, _, _ = cm
    return ((TP + eps) / (TP + FP + eps)).mean()

def recall(cm, eps=1e-7):
    TP, _, _, FN = cm
    return ((TP + eps) / (TP + FN + eps)).mean()

def sensitivity(cm, eps=1e-7):
    return recall(cm, eps)

def specificity(cm, eps=1e-7):
    _, FP, TN, _ = cm
    return ((TN + eps) / (TN + FP + eps)).mean()

def f1_score(cm, eps=1e-7):
    p, r = precision(cm, eps), recall(cm, eps)
    return 2 * p * r / (p + r)

def roc_auc(input: Tensor, target: Tensor, threshold: float = 0.5):
    input = (torch.sigmoid(input) > threshold).to(torch.float32)
    input, target = input.view(-1).cpu().numpy(), target.view(-1).cpu().numpy()
    fpr, tpr, _ = roc_curve(target, input)
    return fpr, tpr, auc(fpr, tpr)

def get_flops_params(model, input_size=(1, 3, 224, 224)):
    from thop import profile

    input = torch.randn(input_size)
    flops, params = profile(model, (input,))
    print('FLOPs: ' + str(flops / 1000 ** 3) + 'G')
    print('Params: ' + str(params / 1000 ** 2) + 'M')