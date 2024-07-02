import math

import torch
from torch import nn
import torch.nn.functional as F

def grad_func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1, device=None):
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

class GradConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
        bias=False
    ):
        super(GradConv2d, self).__init__()
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
        self.gradconv = grad_func

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # 初始化权重
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.gradconv(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, device=self.weight.device)

class GradConvBlock(nn.Module):
    def __init__(self, inplane, ouplane, stride=1):
        super(GradConvBlock, self).__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = GradConv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

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
