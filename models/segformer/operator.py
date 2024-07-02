import math
from functools import partial
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

def hog_gd_func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1, device=None):
    assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
    assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
    assert padding == dilation, 'padding for ad_conv set wrong'
    smooth = 1e-7

    list_x = []
    for i in range(weights.shape[1]):
        list_x.append(torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], device=device))

    list_x = torch.stack(list_x, 0)

    list_y = []
    for i in range(weights.shape[1]):
        list_y.append(torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], device=device))

    list_y = torch.stack(list_y, 0)
    weight_x = torch.mul(weights, list_x)
    weight_y = torch.mul(weights, list_y)

    input_x = F.conv2d(x, weight_x, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    input_y = F.conv2d(x, weight_y, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    input_x = torch.mul(input_x, input_x)
    input_y = torch.mul(input_y, input_y)

    result = torch.add(input_x, input_y) + smooth
    result = result.sqrt()
    return result

def hog_cygd_func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1, device=None):
    assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
    assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
    assert padding == dilation, 'padding for ad_conv set wrong'
    smooth = 1e-7
    shape = weights.shape
    weights = weights.view(shape[0], shape[1], -1)

    weight_x = (weights[:, :, [2, 0, 1, 5, 3, 4, 8, 6, 7]] - weights).view(shape)  # clock-wise
    weight_y = (weights[:, :, [6, 7, 8, 0, 1, 2, 3, 4, 5]] - weights).view(shape)  # clock-wise

    input_x = F.conv2d(x, weight_x, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    input_y = F.conv2d(x, weight_y, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    input_x = torch.mul(input_x, input_x)
    input_y = torch.mul(input_y, input_y)

    result = torch.add(input_x, input_y) + smooth
    result = result.sqrt()

    return result

def lbp_func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1, device=None):
    assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
    assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
    padding = 2 * dilation

    shape = weights.shape
    if weights.is_cuda:
        buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
    else:
        buffer = torch.zeros(shape[0], shape[1], 5 * 5)
    weights = weights.view(shape[0], shape[1], -1)
    buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
    buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
    buffer[:, :, 12] = 0
    buffer = buffer.view(shape[0], shape[1], 5, 5)
    y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return y

class Conv2d(nn.Module):
    def __init__(
        self, func, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
        bias=False
    ):
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.gradconv = func

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # 初始化权重
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.gradconv(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, device=self.weight.device)

class ConvBlock(nn.Module):
    def __init__(self, func, in_channels, ouplane=None, stride=1):
        super(ConvBlock, self).__init__()
        self.stride = stride
        ouplane = ouplane or in_channels
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(in_channels, ouplane, kernel_size=1, padding=0)
        self.conv1 = Conv2d(func, in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class ChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: int = 16,
        **__,
    ):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = x * self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7, **__):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = x * self.sigmoid(out)
        return out

def flatten(seq):
    flatten_seq = []
    for element in seq:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            for sub in flatten(element):
                flatten_seq.append(sub)
        else:
            flatten_seq.append(element)
    return flatten_seq

__factory = {
    "ca": ChannelAttention,
    "sa": SpatialAttention,
    "hog": partial(ConvBlock, func=hog_cygd_func),
    "lbp": partial(ConvBlock, func=lbp_func),
    "identity": nn.Identity,
}

def build(name, in_channels: int):
    return __factory[name](in_channels=in_channels)

modules_1 = {
    "4": ["ca", "sa", "hog", "lbp"],
    "8": flatten([["ca"] * 2, ["sa"] * 2, ["hog"] * 2, ["lbp"] * 2]),
    "12": flatten([["ca"] * 3, ["sa"] * 3, ["hog"] * 3, ["lbp"] * 3]),
    "16": flatten([["ca"] * 4, ["sa"] * 4, ["hog"] * 4, ["lbp"] * 4]),
}

modules_2 = {
    "4": ["ca", "sa", "hog", "lbp"],
    "8": flatten([[["ca", "sa"], ["hog", "lbp"]] * 2]),
    "12": flatten([[["ca", "sa"], ["hog", "lbp"]] * 3]),
    "16": flatten([[["ca", "sa"], ["hog", "lbp"]] * 4]),
}

modules_3 = {
    "4": (t := ["ca", "sa", "hog", "lbp"]),
    "8": flatten([[[e, "identity"] for e in t]]),
    "12": flatten([[[e, "identity", e] for e in t]]),
    "16": flatten([[[e, "identity", e, "identity"] for e in t]]),
}

if __name__ == '__main__':
    # x = torch.randn(1, 32, 16, 16)
    # module = build("8", 32)
    # print(module(x).shape)
    print(modules_3["16"])
