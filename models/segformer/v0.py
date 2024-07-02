from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from einops.layers.torch import Rearrange
from loguru import logger

from models.segformer.backbone import ENCODERS

class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34()

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ))
        self.blocks.append(nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        ))
        self.blocks.append(resnet.layer2)
        self.blocks.append(resnet.layer3)
        self.blocks.append(resnet.layer4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out

class PyramidFeatures(nn.Module):
    def __init__(
        self,
        sa_encoder_name: str = "b0",
        cnn_pyramid_fm: tuple[int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.rnn_encoder = ResNet34()

        segformer_encoder = ENCODERS[sa_encoder_name]()
        self.sa_pyramid_fm = segformer_encoder.embed_dims
        self.sa_layer0 = segformer_encoder.block1
        self.sa_norm_0 = segformer_encoder.norm1
        self.sa_layer1 = segformer_encoder.block2
        self.sa_norm_1 = segformer_encoder.norm2
        self.sa_layer2 = segformer_encoder.block3
        self.sa_norm_3 = segformer_encoder.norm3
        self.sa_layer3 = segformer_encoder.block4
        self.sa_norm_3 = segformer_encoder.norm4
        self.sa_down0 = segformer_encoder.downsamples[0]
        self.sa_down1 = segformer_encoder.downsamples[1]
        self.sa_down2 = segformer_encoder.downsamples[2]

        self.p1_ch = nn.Conv2d(cnn_pyramid_fm[0], self.sa_pyramid_fm[0], kernel_size=1)
        self.p2_ch = nn.Conv2d(cnn_pyramid_fm[1], self.sa_pyramid_fm[1], kernel_size=1)
        self.p3_ch = nn.Conv2d(cnn_pyramid_fm[2], self.sa_pyramid_fm[2], kernel_size=1)
        self.p4_ch = nn.Conv2d(cnn_pyramid_fm[3], self.sa_pyramid_fm[3], kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.rnn_encoder(x)

        fm1 = features[1]  # [b, 64, 56, 56]
        fm1_ch = self.p1_ch(fm1)  # [b, 96, 56, 56]
        fm1_reshaped = Rearrange("b c h w -> b h w c")(fm1_ch)  # [b, 56, 56, 96]
        sw1 = self.sa_layer0(fm1_reshaped)  # [b, 56, 56, 96]
        fm1_sw1_skipped = fm1_reshaped + sw1
        fm1_sw1 = self.sa_down0(fm1_sw1_skipped)  # [b, 28, 28, 192]

        fm1_sw1_d = self.sa_layer1(fm1_sw1)  # [b, 28, 28, 192]
        fm2 = features[2]  # [b, 128, 28, 28]
        fm2_ch = self.p2_ch(fm2)  # [b, 192, 28, 28]
        fm2_reshaped = Rearrange("b c h w -> b h w c")(fm2_ch)  # [b, 28, 28, 192]
        fm2_sw2_skipped = fm2_reshaped + fm1_sw1_d  # [b, 28, 28, 192]
        fm2_sw2 = self.sa_down1(fm2_sw2_skipped)  # [b, 14, 14, 384]

        fm2_sw2_d = self.sa_layer2(fm2_sw2)  # [b, 14, 14, 384]
        fm3 = features[3]  # [b, 256, 14, 14]
        fm3_ch = self.p3_ch(fm3)  # [b, 384, 14, 14]
        fm3_reshaped = Rearrange("b c h w -> b h w c")(fm3_ch)
        fm3_sw3_skipped = fm3_reshaped + fm2_sw2_d  # [b, 14, 14, 384]
        fm3_sw3 = self.sa_down2(fm3_sw3_skipped)  # [b, 7, 7, 768]

        fm3_sw3_d = self.sa_layer3(fm3_sw3)  # [b, 7, 7, 768]
        fm4 = features[4]  # [b, 512, 7, 7]
        fm4_ch = self.p4_ch(fm4)  # [b, 768, 7, 7]
        fm4_reshaped = Rearrange("b c h w -> b h w c")(fm4_ch)
        fm4_sw4_skipped = fm4_reshaped + fm3_sw3_d  # [b, 7, 7, 768]

        return (features[0],
                fm1_sw1_skipped.permute(0, 3, 1, 2),
                fm2_sw2_skipped.permute(0, 3, 1, 2),
                fm3_sw3_skipped.permute(0, 3, 1, 2),
                fm4_sw4_skipped.permute(0, 3, 1, 2))
